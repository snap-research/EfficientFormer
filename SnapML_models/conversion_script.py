import argparse
import torch

import numpy as np
import onnx
from SnapMLEfficientFormer import *
from EfficientFormer import *

def find_layer_idx(graph, initializer):
    for i, node in enumerate(graph.node):
        if initializer in node.input:
            # print(node)
            return i
    print("initializer not found!")
    return -1

def replace_initializer_with_constant_node(graph, layer_idx, name, values, dims):
    # Append an empty node and the end of the graph
    # and move back all subsequent nodes to create space to insert constant node
    graph.node.add()

    for idx in reversed(range(len(graph.node))):
        if idx > layer_idx:
            graph.node.remove(graph.node[idx])
            graph.node.insert(idx, graph.node[idx-1])

    # insert constant node
    const_node = onnx.helper.make_node(
        'Constant',
        inputs=[],
        outputs=[name],
        value=onnx.helper.make_tensor(
            name=name,
            data_type=onnx.TensorProto.FLOAT,
            dims=dims,
            vals=values.flatten().astype(float),
        ),
        name=name
    )

    graph.node.remove(graph.node[layer_idx])
    graph.node.insert(layer_idx, const_node)

    # replace Add node with constant input
    prev_add_node = graph.node[layer_idx+1]
    new_add_node = onnx.helper.make_node(
        prev_add_node.op_type,
        inputs=prev_add_node.input,
        outputs=prev_add_node.output,
        name=prev_add_node.name
    )

    graph.node.remove(graph.node[layer_idx+1])
    graph.node.insert(layer_idx+1, new_add_node)

def transform(model):
    onnx_model = onnx.load(model)
    graph = onnx_model.graph

    # iterate through all initializers
    initializers_to_remove = []
    for initializer in graph.initializer:
        name = initializer.name
        raw_data = initializer.raw_data
        float_data = np.frombuffer(raw_data, np.float32)
        dims = initializer.dims
        layer_idx = find_layer_idx(graph, name)
        if graph.node[layer_idx].op_type in ["Add", "Mul"]:
            replace_initializer_with_constant_node(graph, layer_idx, name, float_data, dims)
            initializers_to_remove.append(initializer)

    for initializer in initializers_to_remove:
        graph.initializer.remove(initializer)
    onnx.checker.check_model(onnx_model)

    onnx.save_model(onnx_model, model)

def pth_to_onnx(filename, num_classes, outputname):
    SnapML_model = SnapML_efficientformer_l1(num_classes=num_classes)

    model_dict["network.6.4.token_mixer.q.weight"] = model_dict["network.6.4.token_mixer.q.weight"].unsqueeze(-1).unsqueeze(-1)
    model_dict["network.6.4.token_mixer.k.weight"] = model_dict["network.6.4.token_mixer.k.weight"].unsqueeze(-1).unsqueeze(-1)
    model_dict["network.6.4.token_mixer.v.weight"] = model_dict["network.6.4.token_mixer.v.weight"].unsqueeze(-1).unsqueeze(-1)
    model_dict["network.6.4.token_mixer.proj.weight"] = model_dict["network.6.4.token_mixer.proj.weight"].unsqueeze(-1).unsqueeze(-1)
    model_dict["network.6.4.mlp.fc1.weight"] = model_dict["network.6.4.mlp.fc1.weight"].unsqueeze(-1).unsqueeze(-1)
    model_dict["network.6.4.mlp.fc2.weight"] = model_dict["network.6.4.mlp.fc2.weight"].unsqueeze(-1).unsqueeze(-1)


    model_dict["head.weight"] = model_dict["head.weight"].unsqueeze(-1).unsqueeze(-1)
    model_dict["dist_head.weight"] = model_dict["dist_head.weight"].unsqueeze(-1).unsqueeze(-1)

    model_dict["network.6.4.token_mixer.attention_biases"] = model_dict["network.6.4.token_mixer.attention_biases"].unsqueeze(1)

    model_dict["network.6.4.token_mixer.proj.weight"] = model_dict["network.6.4.token_mixer.proj.weight"] * model_dict["network.6.4.layer_scale_1"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    model_dict["network.6.4.token_mixer.proj.bias"] = model_dict["network.6.4.token_mixer.proj.bias"] * model_dict["network.6.4.layer_scale_1"]
    model_dict['network.6.4.mlp.fc2.weight'] = model_dict['network.6.4.mlp.fc2.weight'] * model_dict["network.6.4.layer_scale_2"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    model_dict['network.6.4.mlp.fc2.bias'] = model_dict['network.6.4.mlp.fc2.bias'] * model_dict["network.6.4.layer_scale_2"]


    SnapML_model.load_state_dict(model_dict)

    input = torch.rand(1,3,224,224)

    SnapML_model.eval()
    print("Saving ONNX")

    torch.onnx.export(SnapML_model,               # model being run
                    input,                         # model input (or a tuple for multiple inputs)
                    outputname,
                    export_params=True,        # store the trained parameter weights inside the model file
                    do_constant_folding=True, 
                    verbose=True,
                    opset_version=12
    )
    print("Converting ONNX")

    # Change filename if necessary
    MODEL_NAME = outputname

    transform(MODEL_NAME)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process EfficientFormer Model .pth.')

    parser.add_argument('--model',
                        help='Path to input model (pth).')

    parser.add_argument('--output',
                        help='Path to output model (onnx).')

    parser.add_argument('--classes',
                        help='Number of classes.', type=int, default=1000)

    args = parser.parse_args()
    print("Loading Model")

    ckpt = torch.load(args.model, map_location=torch.device('cpu'))
    
    if 'state_dict' in ckpt:
        model_dict = ckpt['state_dict']
    elif 'model' in ckpt:
        model_dict = ckpt['model']
    else:
        model_dict = ckpt

    pth_to_onnx(args.model, args.classes, args.output)
    
    print("DONE")