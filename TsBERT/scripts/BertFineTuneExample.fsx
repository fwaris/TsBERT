#load "packages.fsx"
open TorchSharp
open TsBERT
open System
open System.IO

let device = if torch.cuda_is_available() then torch.CUDA else torch.CPU
printfn $"torch device is %A{device}"

//create model
let cfg = BertConfig.Default //uncased small bert config

//read pretrained weights downloaded from:
//https://tfhub.dev/google/small_bert/bert_uncased_L-2_H-128_A-2/2 
let bertCheckpointFolder = @"C:\temp\bert\variables"
let tensors = TfCheckpoint.CheckpointReader.readCheckpoint bertCheckpointFolder |> Seq.toArray

//data downloaded from
//https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv
let foldr = @"C:\temp\yelp_review_polarity_csv\yelp_review_polarity_csv"
let testCsv = Path.Combine(foldr,"test.csv")
let trainCsv = Path.Combine(foldr,"train.csv")
if File.Exists testCsv |> not then failwith $"File not found; path = {testCsv}"
printfn "%A" trainCsv

type YelpCsv = FSharp.Data.CsvProvider< Sample="a,b", HasHeaders=false, Schema="Label,Text">
type [<CLIMutable>] YelpReview = {Label:int; Text:string}
//need to make labels 0-based so subtract 1
let testSet = YelpCsv.Load(testCsv).Rows |> Seq.map (fun r-> {Label=int r.Label - 1; Text=r.Text}) |> Seq.toArray 
let trainSet = YelpCsv.Load(trainCsv).Rows |> Seq.map (fun r->{Label=int r.Label - 1; Text=r.Text}) |> Seq.toArray

let classes = trainSet |> Seq.map (fun x->x.Label) |> set
let TGT_LEN = classes.Count |> int64

let BATCH_SIZE = 128
let trainBatches = trainSet |> Seq.chunkBySize BATCH_SIZE
let testBatches  = testSet  |> Seq.chunkBySize BATCH_SIZE

open BERTTokenizer
let vocabFile = @"C:\temp\bert\assets\vocab.txt"
let vocab = Vocabulary.loadFromFile vocabFile

let position_ids = torch.arange(cfg.MaxPosEmbeddings).expand(int64 BATCH_SIZE,-1L).``to``(device)

//convert a batch to input and output (X, Y) tensors
let toXY (batch:YelpReview[]) = 
    let xs = batch |> Array.map (fun x-> Featurizer.toFeatures vocab true (int cfg.MaxPosEmbeddings) x.Text "")
    let d_tkns      = xs |> Seq.collect (fun f -> f.InputIds )  |> Seq.toArray
    let d_tkn_typs  = xs |> Seq.collect (fun f -> f.SegmentIds) |> Seq.toArray
    let tokenIds = torch.tensor(d_tkns,     dtype=torch.int).view(-1L,cfg.MaxPosEmbeddings)        
    let sepIds   = torch.tensor(d_tkn_typs, dtype=torch.int).view(-1L,cfg.MaxPosEmbeddings)
    let Y = torch.tensor(batch |> Array.map (fun x->x.Label), dtype=torch.int64).view(-1L)
    (tokenIds,sepIds),Y

//wrap the basic bert model for classification fine tuning
type BertClassification(cfg) as this = 
    inherit torch.nn.Module("BertClassification")

    let bert = new BertModel(cfg)
    let proj = torch.nn.Linear(cfg.Hidden,TGT_LEN)

    do
        this.RegisterComponents()
        this.LoadBertPretrained()

    member _.LoadBertPretrained() =
        BertWeights.loadWeights bert tensors BertWeights.PREFIX (int cfg.EncoderLayers) BertWeights.bertNameMap
    
    member _.forward(tknIds,sepIds,pstnIds) =
        use encoded = bert.forward(tknIds,sepIds,pstnIds)
        encoded --> proj 

//training and evaluation
let _model = new BertClassification(cfg)
_model.``to``(device)
let _loss = torch.nn.CrossEntropyLoss()
let mutable EPOCHS = 1
let mutable verbose = true
let gradCap = 0.1f
let gradMin,gradMax = (-gradCap).ToScalar(),  gradCap.ToScalar()
let opt = torch.optim.Adam(_model.parameters (), 0.001, amsgrad=true)       

let class_accuracy (y:torch.Tensor) (y':torch.Tensor) =
    use i = y'.argmax(1L)
    let i_t = Tensor.getData<int64>(i)
    let m_t = Tensor.getData<int64>(y)
    Seq.zip i_t m_t 
    |> Seq.map (fun (a,b) -> if a = b then 1.0 else 0.0) 
    |> Seq.average

//adjustment for end of data when full batch may not be available
let adjPositions currBatchSize = if int currBatchSize = BATCH_SIZE then position_ids else torch.arange(cfg.MaxPosEmbeddings).expand(currBatchSize,-1L).``to``(device)

let dispose ls = ls |> List.iter (fun (x:IDisposable) -> x.Dispose())

//run a batch through the model; return true output, predicted output and loss tensors
let processBatch ((tkns:torch.Tensor,typs:torch.Tensor), y:torch.Tensor) =
    use tkns_d = tkns.``to``(device)
    use typs_d = typs.``to``(device)
    let y_d    = y.``to``(device)            
    let pstns  = adjPositions tkns.shape.[0]
    if device <> torch.CPU then //these were copied so ok to dispose old tensors
        dispose [tkns; typs; y]
    let y' = _model.forward(tkns_d,typs_d,pstns)
    let loss = _loss.forward(y', y_d)   
    y_d,y',loss

//evaluate on test set; return cross-entropy loss and classification accuracy
let evaluate e =
    _model.eval()
    let lss =
        testBatches 
        |> Seq.map toXY
        |> Seq.map (fun batch ->
            let y,y',loss = processBatch batch
            let ls = loss.ToDouble()
            let acc = class_accuracy y y'            
            dispose [y;y';loss]
            GC.Collect()
            ls,acc)
        |> Seq.toArray
    let ls  = lss |> Seq.averageBy fst
    let acc = lss |> Seq.averageBy snd
    ls,acc

let mutable e = 0

let tks = new System.Threading.CancellationTokenSource()

let train () =
    
    while e < EPOCHS && tks.IsCancellationRequested |> not do
        e <- e + 1
        _model.train()
        let losses = 
            trainBatches 
            |> Seq.choose(fun b -> if tks.IsCancellationRequested then None else Some b)
            |> Seq.map toXY
            |> Seq.mapi (fun i batch ->                 
                opt.zero_grad ()   
                let y,y',loss = processBatch batch
                let ls = loss.ToDouble()  
                loss.backward()
                _model.parameters() |> Seq.iter (fun t -> t.grad().clip(gradMin,gradMax) |> ignore)                            
                use  t_opt = opt.step ()
                if verbose && i % 100 = 0 then
                    let acc = class_accuracy y y'
                    printfn $"Epoch: {e}, minibatch: {i}, ce: {ls}, accuracy: {acc}"                            
                dispose [y;y';loss]
                GC.Collect()
                ls)
            |> Seq.toArray

        let evalCE,evalAcc = evaluate e
        printfn $"Epoch {e} train: {Seq.average losses}; eval acc: {evalAcc}"

    printfn "Done train"

let runner () = async { do train () } 

Async.Start(runner(),tks.Token)

(*
tks.Cancel()   // stop training at end of current minibatch
*)
