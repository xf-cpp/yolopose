import os
import numpy as np
import cv2
import argparse
import onnxruntime
from tqdm import tqdm
from pathlib import Path
from utils.general import non_max_suppression_export,non_max_suppression
import onnx
parser = argparse.ArgumentParser()
# parser.add_argument("--model-path", type=str, default="yolov5s6_pose_640s.onnx")
parser.add_argument("--model-path", type=str, default="yolov5s6_pose_640(3,57).onnx")
parser.add_argument("--img-path", type=str, default="./sample_ips.txt")
parser.add_argument("--dst-path", type=str, default="./sample_ops_onnxrt")
args = parser.parse_args()

#RGB
_CLASS_COLOR_MAP = [
    (0, 0, 255) , # Person (blue).
    (255, 0, 0) ,  # Bear (red).
    (0, 255, 0) ,  # Tree (lime).
    (255, 0, 255) ,  # Bird (fuchsia).
    (0, 255, 255) ,  # Sky (aqua).
    (255, 255, 0) ,  # Cat (yellow).
]
#调色板
palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
            [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
            [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]

pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
radius = 5

def read_img(img_file, img_mean=127.5, img_scale=1/127.5):
    #读取图片，Cv2读取的彩色图片形式为[height,width,3]，其中维度=2上的通道模式通常为BGR,[:,:,::-1]将图片通道转换为RGB
    img = cv2.imread(img_file)[:, :, ::-1]
    #将图片大小调整为640*640，缩放时采用线性差值INTER_LINEAR来保证图像在大小变换中保持平滑
    img = cv2.resize(img, (640,640), interpolation=cv2.INTER_LINEAR)
    #对图像进行归一化 将图像中每个像素减去IMG_MEAN 再除以img_scale 通常这步将图像的像素值缩放到一定范围内
    img = (img - img_mean) * img_scale
    #将图像数据转化为numpy数组，确保数据的数据类型为32位浮点数
    img = np.asarray(img, dtype=np.float32)
    #在数组的第一个维度上添加一个维度 将图像变成Batch维度中的一个数据
    img = np.expand_dims(img,0)
    #将数组维度重新排列，将其变为B C H W的维度
    img = img.transpose(0,3,1,2)
    return img


def model_inference(model_path=None, input=None):
    #onnx_model = onnx.load(args.model_path)
    #model_path指向一个onnx格式的权重，onnxruntime.InferenceSession创建了一个ONNX Runtime的推理会话 会话加载指定的ONNX模型，准备推理
    session = onnxruntime.InferenceSession(model_path, None)
    #get_inputs函数获取模型的输入文件名
    input_name = session.get_inputs()[0].name

    #将输出存储在output中
    '''
        run(output_names, input_feed, run_options=None)
         作用：计算预测的输出        
         参数:
            output_names – 输出的名称，是一个列表，若为空，则不返回任何特定的张量
            input_feed – 字典 { input_name: input_value }
                ONNX模型通常有一个或者多个输入张量，这些张量在模型推理过程中需要被填充
            run_options – See onnxruntime.RunOptions.
         返回值:
            结果列表, 每个结果要不是一个numpy数组，要不是一个稀疏张量，列表或者字典      
    '''
    output = session.run([], {input_name: input})
    return output


def model_inference_image_list(model_path, img_path=None, mean=None, scale=None, dst_path=None):
    #创建保存图像的路径，如果存在则忽略
    os.makedirs(args.dst_path, exist_ok=True)
    #打开需要处理的图像文件路径，并转化为list
    img_file_list = list(open(img_path))
    #返回一个元祖 每个元祖包含一个序列和迭代器的值
    pbar = enumerate(img_file_list)
    max_index = 20
    #设置进度条
    pbar = tqdm(pbar, total=min(len(img_file_list), max_index))

    for img_index, img_file  in pbar:
        #设置进度条
        pbar.set_description("{}/{}".format(img_index, len(img_file_list)))
        #去除字符串img_file末尾的空格或空字符
        img_file = img_file.rstrip()
        #输入图像数据
        input = read_img(img_file, mean, scale)
        #输出模型数据
        output = model_inference(model_path, input)
        #在dst_path中保存img_file文件 os.path.basename提取路径中的文件名
        dst_file = os.path.join(dst_path, os.path.basename(img_file))
        #后处理文件
        post_process(img_file, dst_file, output[0], score_threshold=0.3)



def post_process(img_file, dst_file, output, score_threshold=0.3):
    """
    Draw bounding boxes on the input image. Dump boxes in a txt file.
    """

    #yolov5s6_pose_640_ti_lite_54p9_82p2.onnx 输出shape为[2,57] yolov5s6_pose_640.onnx 输出shape为[1.3.80.80.57]
    # [2,57]-> 57 = x_min ,y_min ,x_max ,y_min ,box_conf ,cls_conf 17 *3  三个通道的关键点
    #将output张量中的值提取出来
    det_bboxes, det_scores, det_labels, kpts = output[:, 0:4], output[:, 4], output[:, 5], output[:, 6:]
    # print("det_bboxes.shape")
    # print(det_bboxes.shape)
    # print('det_scores.shape')
    # print(det_scores.shape)
    # print('det_labels.shape')
    # print(det_labels.shape)
    # print('kpts.shape')
    # print(kpts.shape)
    #输入提取测试图像
    img = cv2.imread(img_file)
    #To generate color based on det_label, to look into the codebase of Tensorflow object detection api.

    path = Path(dst_file)
    suffix = path.suffix

    dst_txt_file = dst_file.replace(suffix, '.txt')

    #准备写入预测框信息的txt文件

    f = open(dst_txt_file, 'wt')
    for idx in range(len(det_bboxes)):
        #len(det_bboxed) = 2
        #读取 box数据
        det_bbox = det_bboxes[idx]
        #读取关键点数据
        kpt = kpts[idx]
        #如果预测值>0 那么将该预测值的 cls_conf box_conf x_min y_min x_max y_max 写入进txt文件
        if det_scores[idx]>0:
            f.write("{:8.0f} {:8.5f} {:8.5f} {:8.5f} {:8.5f} {:8.5f}\n".format(det_labels[idx], det_scores[idx], det_bbox[1], det_bbox[0], det_bbox[3], det_bbox[2]))
        #如果预测置信度值大于置信度阈值
        if det_scores[idx]>score_threshold:
            #选择指定的颜色
            color_map = _CLASS_COLOR_MAP[int(det_labels[idx])]

            '''
            作用：
                在指定的图片上画框
                 cv2.rectangle(image, start_point, end_point, color, thickness)
                参数：
                    image：指定画框的图片
                    start_point:框的开始点，一般用一个元祖(x,y)表示
                    end_point:框的结束点，一般用一个元祖(x,y)表示
                    color：框的颜色 用BGR表示
                    thickness：用多少像素(px)画框，如果=-1，用指定的颜色填满画的框
            '''
            # 在图像上画框,cv2.rectangle根据预测框的左上和右下角坐标进行画框，color_map[::-1]表示颜色用GBR表示画出，2表示检测框的宽度为2像素

            img = cv2.rectangle(img, (int(det_bbox[0]), int(det_bbox[1])), (int(det_bbox[2]), int(det_bbox[3])), color_map[::-1], 2)
            #在图像上写字，要写的字 写的字体的左上角坐标 字体 字体大小 颜色 宽度
            cv2.putText(img, "id:{}".format(int(det_labels[idx])), (int(det_bbox[0]+5),int(det_bbox[1])+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            cv2.putText(img, "score:{:2.1f}".format(det_scores[idx]), (int(det_bbox[0] + 5), int(det_bbox[1]) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_map[::-1], 2)
            #在图像上画关键点
            plot_skeleton_kpts(img, kpt)
    #将画好框的img写入指定文件
    cv2.imwrite(dst_file, img)
    f.close()

def plot_skeleton_kpts(im, kpts, steps=3):
    #每个通道层的关键点数 = 总的kpts /层数 -> num_kpts = 51 /3
    num_kpts = len(kpts) // steps
    #plot keypoints
    #一共画17个点 kid = 0-16
    for kid in range(num_kpts):
        #取rgb三通道颜色值
        r, g, b = pose_kpt_color[kid]
        #取关键点的坐标 x y
        x_coord, y_coord = kpts[steps * kid], kpts[steps * kid + 1]
        #取关键点的置信度 conf
        conf = kpts[steps * kid + 2]
        #如果关键点置信度大于0.5 则将关键点画在图像上
        if conf > 0.5: #Confidence of a keypoint has to be greater than 0.5
            #根据中心点在图像上画圆
            #需要画的图像/圆点坐标/半径/圆圈的颜色/圆圈宽度 -1表示用指定颜色填满圆圈
            cv2.circle(im, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)
    #plot skeleton
    #画出关键点之间的骨架
    for sk_id, sk in enumerate(skeleton):
        '''
        palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])

        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12],
                    [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3],
                    [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        
        pose_limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        pose_kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        radius = 5
    '''
        #指定骨架连线的颜色
        r, g, b = pose_limb_color[sk_id]
        #-1是为了适配元祖的下标 取出两个关键点之间的坐标
        pos1 = (int(kpts[(sk[0]-1)*steps]), int(kpts[(sk[0]-1)*steps+1]))
        pos2 = (int(kpts[(sk[1]-1)*steps]), int(kpts[(sk[1]-1)*steps+1]))
        conf1 = kpts[(sk[0]-1)*steps+2]
        conf2 = kpts[(sk[1]-1)*steps+2]
        if conf1>0.5 and conf2>0.5: # For a limb, both the keypoint confidence must be greater than 0.5
            cv2.line(im, pos1, pos2, (int(r), int(g), int(b)), thickness=2)


def main():
    model_inference_image_list(model_path=args.model_path, img_path=args.img_path,
                               mean=0.0, scale=0.00392156862745098,
                               dst_path=args.dst_path)

if __name__== "__main__":
    main()
