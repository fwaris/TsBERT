namespace TsBERT
open TorchSharp

type EncoderActivation = Gelu | Relu

type BertConfig = 
    {
        Hidden              : int64
        VocabularySize      : int64
        TokenTypes          : int64
        MaxPosEmbeddings    : int64
        EpsLayerNorm        : float
        NHeads              : int64
        AttnDropoutProb     : float
        HiddenDropoutProb   : float
        EncoderLayers       : int64
        EncoderActivation   : EncoderActivation
    }
    with
    static member Default =
        {
            Hidden              = 128L
            VocabularySize      = 30522L
            TokenTypes          = 2L
            MaxPosEmbeddings    = 512L
            EpsLayerNorm        = 1e-12
            NHeads              = 2L
            AttnDropoutProb     = 0.1
            HiddenDropoutProb   = 0.1
            EncoderLayers       = 2L
            EncoderActivation   = EncoderActivation.Gelu        
        }

type BertEmbedding(cfg) as this = 
    inherit torch.nn.Module<torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor>("embeddings")
    
    let word_embeddings         = torch.nn.Embedding(cfg.VocabularySize,cfg.Hidden,padding_idx=0L)
    let position_embeddings     = torch.nn.Embedding(cfg.MaxPosEmbeddings,cfg.Hidden)
    let token_type_embeddings   = torch.nn.Embedding(cfg.TokenTypes,cfg.Hidden)
    let LayerNorm               = torch.nn.LayerNorm([|cfg.Hidden|],cfg.EpsLayerNorm)
    let dropout                 = torch.nn.Dropout(cfg.HiddenDropoutProb)

    do
        this.RegisterComponents()

    override _.forward(input_ids:torch.Tensor, token_type_ids:torch.Tensor, position_ids:torch.Tensor) =   
    
        let embeddings =      
            (input_ids       --> word_embeddings)        +
            (token_type_ids  --> token_type_embeddings)  +
            (position_ids    --> position_embeddings)

        embeddings --> LayerNorm --> dropout             // the --> operator works for simple 'forward' invocations


type BertPooler(cfg) as this = 
    inherit torch.nn.Module<torch.Tensor,torch.Tensor>("pooler")

    let dense = torch.nn.Linear(cfg.Hidden,cfg.Hidden)
    let activation = torch.nn.Tanh()

    let ``:`` = torch.TensorIndex.Colon
    let first = torch.TensorIndex.Single(0L)

    do 
        this.RegisterComponents()

    override _.forward (hidden_states) =
        let first_token_tensor = hidden_states.index(``:``, first) //take first token of the sequence as the pooled value
        first_token_tensor --> dense --> activation

type BertModel(cfg) as this =
    inherit torch.nn.Module("bert")

    let embeddings = new BertEmbedding(cfg)
    let pooler = new BertPooler(cfg)

    let encoderLayer = torch.nn.TransformerEncoderLayer(cfg.Hidden, 
                                                        cfg.NHeads, 
                                                        cfg.MaxPosEmbeddings, 
                                                        cfg.AttnDropoutProb, 
                                                        activation=match cfg.EncoderActivation with Gelu -> torch.nn.Activations.GELU | Relu -> torch.nn.Activations.ReLU)

    let encoder = torch.nn.TransformerEncoder(encoderLayer, cfg.EncoderLayers)

    do 
        this.RegisterComponents()

    member this.forward(input_ids:torch.Tensor, token_type_ids:torch.Tensor, position_ids:torch.Tensor,?mask:torch.Tensor) =
        let src = embeddings.forward(input_ids, token_type_ids, position_ids)
        let srcBatchDim2nd = src.permute(1L,0L,2L) //PyTorch transformer requires input as such. See the Transformer docs
        let encoded = match mask with None -> encoder.forward(srcBatchDim2nd) | Some mask -> encoder.forward(srcBatchDim2nd,mask)
        let encodedBatchFst = encoded.permute(1L,0L,2L)
        encodedBatchFst --> pooler