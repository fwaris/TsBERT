namespace TsBERT
open System
open TorchSharp

module Tensor = 
    //Note: ensure 't matches tensor datatype otherwise ToArray might crash the app (i.e. exception cannot be caught)
    let private _getData<'t when 't:>ValueType and 't:struct and 't : (new:unit->'t) > (t:torch.Tensor) =
        let s = t.data<'t>()
        s.ToArray()

    let getData<'t when 't:>ValueType and 't:struct and 't : (new:unit->'t)>  (t:torch.Tensor) =
        if t.device_type <> DeviceType.CPU then 
            //use t1 = t.clone()
            use t2 = t.cpu()
            _getData<'t> t2
        else 
            _getData<'t> t
  
    let setData<'t when 't:>ValueType and 't:struct and 't : (new:unit->'t)> (t:torch.Tensor) (data:'t[]) =
        if t.device_type = DeviceType.CPU |> not then failwith "tensor has to be on cpu for setData"        
        let s = t.data<'t>()
        s.CopyFrom(data,0,0L)

module BertWeights = 
    type PostProc = V | H | T | N

    let postProc (ts:torch.Tensor list) = function
        | V -> torch.vstack(ResizeArray ts)
        | H -> torch.hstack(ResizeArray ts)
        | T -> ts.Head.T                  //Linear layer weights need to be transformed. See https://github.com/pytorch/pytorch/issues/2159
        | N -> ts.Head

    let bertNameMap =
        [
            "encoder.layers.#.self_attn.in_proj_weight",["encoder/layer_#/attention/self/query/kernel"; 
                                                         "encoder/layer_#/attention/self/key/kernel";    
                                                         "encoder/layer_#/attention/self/value/kernel"],        V

            "encoder.layers.#.self_attn.in_proj_bias",  ["encoder/layer_#/attention/self/query/bias";
                                                         "encoder/layer_#/attention/self/key/bias"; 
                                                         "encoder/layer_#/attention/self/value/bias"],          H

            "encoder.layers.#.self_attn.out_proj.weight", ["encoder/layer_#/attention/output/dense/kernel"],    N
            "encoder.layers.#.self_attn.out_proj.bias",   ["encoder/layer_#/attention/output/dense/bias"],      N


            "encoder.layers.#.linear1.weight",          ["encoder/layer_#/intermediate/dense/kernel"],          T
            "encoder.layers.#.linear1.bias",            ["encoder/layer_#/intermediate/dense/bias"],            N

            "encoder.layers.#.linear2.weight",          ["encoder/layer_#/output/dense/kernel"],                T
            "encoder.layers.#.linear2.bias",            ["encoder/layer_#/output/dense/bias"],                  N

            "encoder.layers.#.norm1.weight",            ["encoder/layer_#/attention/output/LayerNorm/gamma"],   N
            "encoder.layers.#.norm1.bias",              ["encoder/layer_#/attention/output/LayerNorm/beta"],    N

            "encoder.layers.#.norm2.weight",            ["encoder/layer_#/output/LayerNorm/gamma"],             N
            "encoder.layers.#.norm2.bias",              ["encoder/layer_#/output/LayerNorm/beta"],              N

            "embeddings.word_embeddings.weight"         , ["embeddings/word_embeddings"]           , N
            "embeddings.position_embeddings.weight"     , ["embeddings/position_embeddings"]       , N
            "embeddings.token_type_embeddings.weight"   , ["embeddings/token_type_embeddings"]     , N
            "embeddings.LayerNorm.weight"               , ["embeddings/LayerNorm/gamma"]           , N
            "embeddings.LayerNorm.bias"                 , ["embeddings/LayerNorm/beta"]            , N
            "pooler.dense.weight"                       , ["pooler/dense/kernel"]                  , T
            "pooler.dense.bias"                         , ["pooler/dense/bias"]                    , N
        ]

    let PREFIX = "bert"
    let addPrefix prefix (s:string) = $"{prefix}/{s}"
    let sub n (s:string) = s.Replace("#",string n)

    open TfCheckpoint

    //create a PyTorch tensor from TF checkpoint tensor data
    let toFloat32Tensor (shpdTnsr:CheckpointReader.ShapedTensor) = 
        match shpdTnsr.Tensor with
        | CheckpointReader.TensorData.TdFloat ds -> torch.tensor(ds, dimensions=ReadOnlySpan shpdTnsr.Shape)
        | _                                      -> failwith "TdFloat expected"
    
    //set the value of a single parameter
    let performMap (tfMap:Map<string,_>) (ptMap:Map<string,Modules.Parameter>) (torchName,tfNames,postProcType) = 
        let torchParm = match ptMap.TryGetValue torchName with true,n -> n | _ -> failwith $"name not found {torchName}"
        let fromTfWts = tfNames |> List.map (fun n -> tfMap.[n] |> toFloat32Tensor) 
        let parmTensor = postProc fromTfWts postProcType
        if torchParm.shape <> parmTensor.shape then failwithf $"Mismatched weights for parameter {torchName}; parm shape: %A{torchParm.shape} vs tensor shape: %A{parmTensor.shape}"
        Tensor.setData<float32> torchParm (Tensor.getData<float32>(parmTensor))
    
    //set the parameter weights of a PyTorch model given checkpoint and nameMap
    let loadWeights (model:torch.nn.Module) checkpoint prefix encoderLayers nameMap =
        let tfMap = checkpoint |> Map.ofSeq
        let ptMap = model.named_parameters() |> Seq.map (fun struct(n,m) -> n,m) |> Map.ofSeq
    
        //process encoder layers
        for l in 0 .. encoderLayers - 1 do
            nameMap
            |> List.filter (fun (p:string,_,_) -> p.Contains("#")) 
            |> List.map (fun (p,tns,postProc) -> sub l p, tns |> List.map ((addPrefix prefix) >> (sub l)), postProc)
            |> List.iter (performMap tfMap ptMap)
    
        nameMap
        |> List.filter (fun (p,_,_) -> p.Contains("#") |> not)
        |> List.map (fun (name,tns,postProcType) -> name, tns |> List.map (addPrefix prefix), postProcType)
        |> List.iter (performMap tfMap ptMap)