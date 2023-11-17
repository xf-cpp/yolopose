import torch.onnx
#根据模型图片输入设定input的h和w的尺寸大小，我是512
input = (torch.randn([1,3,640,640]).cuda())
input_names = ['input_1']
output_names = ['output']

weight_path = './yolov5s6_pose_640.pt'
model=torch.load(weight_path)
model = model.cuda()
model.eval()
print('load model over')

torch.onnx.export(model,
                  args=input,
                  f = './onnx/*****.onnx',
                  opset_version=11,
                  input_names=input_names,
                  output_names=output_names
                  )
print('to onnx success!')
