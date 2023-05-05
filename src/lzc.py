import numpy as np
import os
import cv2
import torchvision.transforms as transforms
from PIL import Image
import timm
import torch
import neural_renderer
import utils.nmr_test as nmr
from PIL import Image

def select_device(device='', batch_size=None, newline=True):
    # device = 'cpu' or '0' or '0,1,2,3'
    device = str(device).strip().lower().replace('cuda:', '')  # to string, 'cuda:0' to '0'
    cpu = device == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'


    return torch.device('cuda:0' if cuda else 'cpu')


def npz_png():
    # 读取.npz文件
    # 存放npz文件路径
    path = r'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages\src\data2\phy_attack\train1'
    image_path = r'E:\BaiduNetdiskDownload\DualAttentionAttack\phy_attack\img'
    label_path = r'E:\BaiduNetdiskDownload\DualAttentionAttack\phy_attack\train\label'
    for i in os.listdir(path):
        # npz文件地址
        npz_path = os.path.join(path, i)
        sampled_batch = np.load(npz_path)
        img = sampled_batch["img"]
        print(sampled_batch['veh_trans'])
        print(sampled_batch['cam_trans'])
        print()
        # cv2.imwrite(os.path.join(path, i)+'.png',cv2.cvtColor(img , cv2.COLOR_RGB2BGR))

def resize():
    # resize
    transform = transforms.Compose([
        transforms.Resize((438, 438)),
        # transforms.CenterCrop((1767 , 1767)),
        # transforms.Resize((421, 421)),
    ])

    path = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\letter_size'
    path1 = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\letter_size'
    # path = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents'
    # path1 = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1'
    for i in os.listdir(path):
        # if i[0] == 'm':
        img_path = os.path.join(path, i)
        img = Image.open(img_path)
        img = transform(img)
        i = 'data' + i
        img1_path = os.path.join(path1, i)
        img.save(img1_path)

def save_npz():
    # 保存.npz
    path = r'F:\new_sign_data\crop_img_npz1'  # 图片路径
    path1 = r'F:\new_sign_data\npz1'
    path2 = r'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages1\src\data2\phy_attack\train'
    img_arr = []
    veh_trans = []
    cam_trans = []
    for i in os.listdir(path):
        img_path = os.path.join(path, i)
        img = Image.open(img_path).convert('RGB')
        img_arr.append(np.array(img, dtype=np.uint8))
    for i in os.listdir(path1):
        # npz文件地址
        npz_path = os.path.join(path1, i)
        sampled_batch = np.load(npz_path)
        veh_trans.append(sampled_batch['veh_trans'])
        cam_trans.append(sampled_batch['cam_trans'])
        # cv2.imwrite(os.path.join(image_path, i)+'.png',img)
    for k in range(len(os.listdir(path1))):
        i = os.listdir(path1)[k]
        npz_path = os.path.join(path2, i)
        np.savez(npz_path , img = img_arr[k], veh_trans = veh_trans[k], cam_trans = cam_trans[k])

def crop_img():
    path = r'F:\carla\carla_sign_img6'
    path1 = r'F:\carla\carla_sign_img5_resize'
    # path = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents'
    # path1 = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1'
    for i in os.listdir(path):
        img_path = os.path.join(path, i)
        img = Image.open(img_path)
        img = img.crop((0, 0, 1080, 1080))
        img1_path = os.path.join(path1, i)
        img.save(img1_path)

def dec_img():
    path = r'F:\new_sign_data\physical_union'  # 图片路径
    model = torch.hub.load(r'D:\PyCharmProject\yolov3-master', 'custom',
                           r'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages\src\yolov3.pt',
                           source='local')  # or yolov3-spp, yolov3-tiny, custom
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5x')         # 0.86  yolov5
    model.eval()
    total = 0.
    succ = 0.
    print("start")
    for i in os.listdir(path):
        total += 1
        img_path = os.path.join(path, i)
        results = model(img_path)
        (results.pandas().xyxy[0]['name'] == 'stop sign').any()
        # print(len(results.pandas().xyxy[0]))
        if (results.pandas().xyxy[0]['name'] == 'stop sign').any():
            succ += 1
        # results.save(r'F:\carla\physics_res\output')
    print(succ / total)

def autoLable():
    # coordinate = torch.nonzero(images[0])
    path = r'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages1\src\data2\phy_attack\masks'
    for i in os.listdir(path):
        img_path = os.path.join(path, i)
        img = cv2.imread(img_path , cv2.IMREAD_GRAYSCALE)
        coordinate = np.nonzero(img)
        x_min = coordinate[1].min()
        x_max = coordinate[1].max()
        y_min = coordinate[0].min()
        y_max = coordinate[0].max()
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)
        # cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        w = x_max - x_min
        h = y_max - y_min
        x_c = (x_min + w/2) / 640
        y_c = (y_min + h/2) / 640
        w /= 640
        h /= 640
        label_path = 'D:/PyCharmProject/Full-coverage-camouflage-adversarial-attack-gh-pages1/src/data2/phy_attack/train_label_new/'
        label = open(label_path + i[:-4] + '.txt', mode='w')
        # print(x_c , y_c , w , h)
        label.write('11 ' + str(x_c) + ' ' + str(y_c) + ' ' + str(w) + ' ' + str(h) + ' ') #\n 换行符
        # break

# 根据masks 来创造 label
def create_txt():
    path = r'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages1\src\data2\phy_attack\train_new'
    for i in os.listdir(path):
        file = open('D:/PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages1/src/data2/phy_attack/train_label_new/' + i[:-4] + '.txt', 'w')
    autoLable()

def cal_texture(texture_param, texture_origin, texture_mask, texture_content=None, CONTENT=False,):
    # 计算纹理
    if CONTENT:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    # return texture_origin
    return texture_origin * (1 - texture_mask) + texture_mask * textures

def gen_tex():
    path = './carassets/new_mm/mask1'

    for i in os.listdir(path):
        if i[-4:] == '.obj':
            texture_size = 70
            filename_obj = os.path.join(path, i)
            vertices, faces, texture_origin = neural_renderer.load_obj(filename_obj=filename_obj,
                                                                       texture_size=texture_size,
                                                                       load_texture=True)
            # texture_param = texture_origin.detach()
            texture_param = np.random.random((1, faces.shape[0], texture_size, texture_size, texture_size, 3)).astype('float32')
            log_dir = 'logs/new_mm'
            # j = i[:-4] + '.npy'
            j = 'test.npy'
            np.save(os.path.join(log_dir, j), texture_param)
            break


def create_Dataset(device):
    texture_size = 70
    vertices, faces, texture_origin = neural_renderer.load_obj(filename_obj='carassets/new_mm/mask/black.obj', texture_size=texture_size,
                                                               load_texture=True)

    # texture_param = texture_origin.detach()
    # log_dir = 'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages\src\logs\mm'
    # np.save(os.path.join(log_dir, 'texture_mask.npy'), texture_param.data.cpu().numpy())
    # return
    # texture_param = np.random.random((1, faces.shape[0], texture_size, texture_size, texture_size, 3)).astype('float32')
    texture_param = np.load('logs/new_mm/test.npy')
    texture_param = torch.from_numpy(texture_param).to(device)
    # print(texture_param)
    # print(torch.max(texture_param))
    # print(torch.min(texture_param))
    # textures = torch.from_numpy(texture_param).to(device)

    texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
    with open('carassets/new_mm/new_mm_faces.txt', 'r') as f:
        face_ids = f.readlines()
        for face_id in face_ids:
            if face_id != '\n':
                texture_mask[int(face_id) - 1, :, :, :,
                :] = 1  # adversarial perturbation only allow painted on specific areas
    texture_mask = torch.from_numpy(texture_mask).to(device).unsqueeze(0)
    textures = cal_texture(texture_param, texture_origin, texture_mask)
    faces = faces[None, :, :]
    vertices = vertices[None, :, :]

    path = r'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages1\src\data2\phy_attack\train'
    mask_renderer = nmr.NeuralRenderer(img_size=640).to(device)
    mask_renderer.renderer.renderer.camera_mode = "look"
    mask_renderer.renderer.renderer.light_direction = [1, 0, 1]
    mask_renderer.renderer.renderer.camera_up = [0, 0, 1]
    mask_renderer.renderer.renderer.background_color = [1, 1, 1]
    # mask_renderer.renderer.renderer.background_color = [0, 0, 0]

    for i in os.listdir(path):
        npz_path = os.path.join(path, i)
        sampled_batch = np.load(npz_path)
        img = sampled_batch['img']
        veh_trans = sampled_batch['veh_trans']
        cam_trans = sampled_batch['cam_trans']
        cam_trans1 = [[2, 0, 0], [0, 180, 0]]
        veh_trans1 = [[0, 0, 0], [0, 0, 0]]
        path1 = i[:-4] + '.png'
        # path2 = int(path1[4:][:-4])
        # if path2 >= 200 and path2 < 300:  # 左
        #     cam_trans[0][1] += 3
        # elif path2 < 100 or path2 >= 300: # 右
        #     cam_trans[0][1] -= 3
        # if path2 >= 300 :
        #     cam_trans[0][1] -= 3

        cam_trans[0][2] -= 0.2

        eye, camera_direction, camera_up = nmr.get_params(cam_trans1, veh_trans1)



        mask_renderer.renderer.renderer.eye = eye
        mask_renderer.renderer.renderer.camera_direction = camera_direction
        mask_renderer.renderer.renderer.camera_up = camera_up

        imgs_pred = mask_renderer.forward(vertices, faces, textures)
        imgs_pred1 = mask_renderer.forward(vertices, faces, textures)

        img2 = 255 * imgs_pred.cpu().detach().numpy()
        im1 = np.transpose(img2[0], (1, 2, 0))
        im1 = Image.fromarray(np.uint8(im1))
        # im1.save('./data2/phy_attack/test_img/7.jpg')
        Image.fromarray(np.uint8(255. * imgs_pred.squeeze().cpu().data.numpy().transpose(1, 2, 0))).show()
        break
        # 生成mask
        img3 = imgs_pred.cpu().detach().numpy()
        im1 = np.transpose(img3[0], (1, 2, 0))
        arr1 = np.ones_like(im1[: , : , 0])
        arr1[(im1[: , : , 0] + im1[: , : , 1] + im1[: , : , 2]) / 3. == 1.] = 0.
        msk = Image.fromarray((arr1 * 255.).astype(np.uint8)).convert('L')
        msk.save('D:/PyCharmProject/Full-coverage-camouflage-adversarial-attack-gh-pages1/src/data2/phy_attack/masks/' + path1)
        # return

        # 生成组合图
        # imgs_pred = imgs_pred / torch.max(imgs_pred)
        # mask_dir = os.path.join('./data2/phy_attack/', 'masks/')
        # mask_file = os.path.join(mask_dir, path1)
        # mask = cv2.imread(mask_file)
        # mask = cv2.resize(mask, (640 , 640))
        # mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
        # mask = torch.from_numpy(mask.astype('float32')).to(device)

        img = img.transpose((2, 0, 1))  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        import random
        # alpha = 0
        # if alpha >= 0:
        #     img = (1 - mask) * img + (255. * imgs_pred * (1 - alpha) + 255. * alpha) * mask
        # else :
        #     img = (1 - mask) * img + (255. * imgs_pred * (1 + alpha)) * mask
        # img = (1 - mask) * img + (255. * imgs_pred) * mask
        img = img + (255. * imgs_pred)


        img1 = img.cpu().detach().numpy()
        im = np.transpose(img1[0], (1, 2, 0))
        im = Image.fromarray(np.uint8(im))
        # im.save(os.path.join('./data2/phy_attack/test_img' , path1))
        # Image.fromarray(np.uint8(img.squeeze().cpu().data.numpy().transpose(1, 2, 0))).show()
        # break

def black_mm():
    bg_path = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\letter_size\datalarge.png'
    bg_img = Image.open(bg_path).convert('RGB')
    # bg_img.save(r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\zhidai\mm3.png')
    bg_img_arr = np.array(bg_img, dtype=np.uint8)
    bg_img_arr[bg_img_arr != 255.] = 0
    bg_img_1 = Image.fromarray(bg_img_arr.astype(np.uint8)).convert('L')
    # bg_img_1.save(r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\zhidai\mm1.png')
    coordinate = np.nonzero(bg_img_arr - 255.)
    x_min = coordinate[1].min()
    x_max = coordinate[1].max()
    y_min = coordinate[0].min()
    y_max = coordinate[0].max()

    bg_img_arr[y_min:y_max , x_min:x_max] = 0
    bg_img_arr[bg_img_arr == 255.] = 0.
    bg_img_arr[y_min:y_max, x_min:x_max] = 255.
    bg_img_2 = Image.fromarray(bg_img_arr.astype(np.uint8)).convert('L')
    bg_img_2.save(r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\letter_size\datalarge1.png')

    # save_path = os.path.join(r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\letter_size' , )
    # bg_img_2.save(save_path)
    # bg_img_arr1 =
    # coordinate = np.nonzero(bg_img_arr)
    # bg_img_arr

def merge_mm():
    bg_path = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\new_mm\2.png'
    bg_img = Image.open(bg_path).convert('RGB')
    bg_img = np.array(bg_img, dtype=np.uint8)

    mask_path = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\letter_size_resize'
    for i in os.listdir(mask_path):
        mask = Image.open(os.path.join(mask_path , i)).convert('L')
        mask = np.array(mask, dtype=np.float32)
        mask = mask[:, :, np.newaxis].repeat(3, axis=2)
        mask /= 255.
        letter = i[:-4]
        # fg_path = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\resize_zimu'
        letter += '.jpg'
        # fg_img = Image.open(os.path.join(fg_path , letter)).convert('RGB')
        # fg_img = np.array(fg_img, dtype=np.float32)
        img = (1 - mask) * ( 0.5 * 255. + bg_img * 0.5) + mask * bg_img
        # fg_img -= 1
        # img = 0.7 * fg_img + bg_img
        img = Image.fromarray(img.astype(np.uint8))
        img_path = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\letter_size_resize'
        img.save(os.path.join(img_path , letter))



    # mask_path = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\new_mm\2.png'
    #
    # bg_path1 = r'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages\src\data2\phy_attack\res\4.png'
    # bg_img1 = Image.open(bg_path1).convert('RGB')
    # h = y_max - y_min
    # w = x_max - x_min
    # bg_img1 = transforms.Resize((h, w), interpolation=Image.BILINEAR)(bg_img1)
    # bg_img_arr2 = np.array(bg_img1, dtype=np.float64)
    # alpha = -0.1
    # bg_img_arr2 *= (1 + alpha)
    #
    # bg_path3 = r'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages\src\data2\phy_attack\res\2.jpg'
    # bg_img3 = Image.open(bg_path3).convert('RGB')
    # bg_img_arr3 = np.array(bg_img3, dtype=np.uint8)
    #
    # bg_img_arr3[y_min : y_max , x_min : x_max] = bg_img_arr2
    # bg_img4 = Image.fromarray(bg_img_arr3.astype(np.uint8)).convert('RGB')
    # bg_img4.save(r'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages\src\data2\phy_attack\res\5.jpg')



def resize_sign_img():
    # resize
    transform = transforms.Compose([
        transforms.CenterCrop(1024),
        transforms.Resize((640, 640))
    ])
    path = r'G:\sign_img\test'
    path1 = r'G:\sign_img\test_resize'
    for i in os.listdir(path):
        img_path = os.path.join(path, i)
        img = Image.open(img_path)
        img = transform(img)
        i = 'data' + i
        img1_path = os.path.join(path1, i)
        img.save(img1_path)

def crop_img():
    path = r'F:\new_sign_data\img'
    path1 = r'F:\new_sign_data\crop_img'
    for i in os.listdir(path):
        img_path = os.path.join(path, i)
        img = cv2.imread(img_path)
        cropped = img[0:1080, 520:1600]  # 裁剪坐标为[y0:y1, x0:x1]
        crop_path = os.path.join(path1, i)
        cv2.imwrite(crop_path, cropped)

def img_rotate():
    path = r'D:\PyCharmProject\DualAttentionAttack-main\src\contents1\mer_img'

    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path, i))
        img = cv2.rotate(img, cv2.ROTATE_180)
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        j = i[:-4] + '_1.jpg'
        cv2.imwrite(os.path.join(path, j), img)

def gen_white():
    data_path = r'F:\new_sign_data\test_lun1'
    for i in os.listdir(data_path):
        patch_path = os.path.join(data_path , i)
        print(patch_path)
        patch = cv2.imread(patch_path)[: , : , ::-1]
        # patch_arr = np.array(patch , dtype = np.float32)
        # patch_arr = (patch_arr[: , : , 0] + patch_arr[: , : , 1] + patch_arr[: , : , 2]) / 3
        # patch_arr = patch_arr[:, :, np.newaxis].repeat(3, axis=2)
        patch_arr_black = np.zeros_like(patch)
        patch_arr_black[patch != 255.] = 255.
        patch_arr_black[patch == 255.] = 0.
        # print(patch_arr_black.shape)

        patch1 = Image.fromarray(patch_arr_black.astype(np.uint8)).convert('RGB')
        # print(i)
        patch1.save(os.path.join(r'F:\new_sign_data\test_lun1_mask' , 'mask_' + i))
        # break
def light():
    data_path = r'F:\new_sign_data\test_lun1'
    mask_path = r'F:\new_sign_data\test_lun1_mask'
    for i in os.listdir(data_path):
        patch_path = os.path.join(data_path , i)
        mask_img_path = os.path.join(mask_path , 'mask_' + i)
        patch = cv2.imread(patch_path)
        mask = cv2.imread(mask_img_path)
        mask = np.logical_or(mask[:, :, 0], mask[:, :, 1], mask[:, :, 2])
        mask = mask.astype('float32')
        mask = mask[:, :, np.newaxis].repeat(3, axis=2)
        import random
        # alpha = random.uniform(-0.3, -0.1)
        alpha = -0.7
        if alpha >= 0:
            patch = (1 - mask) * patch + (patch * (1 - alpha) + 255. * alpha) * mask
        else:
            patch = (1 - mask) * patch + (patch * (1 + alpha)) * mask
        # print(mask.shape)

        cv2.imwrite(os.path.join(r'F:\new_sign_data\test_lun_res', '1_' + i[:-4] + '.jpg'), patch)


def union_img():
    v5_path = r'F:\new_sign_data\physical_v5'
    v7_path = r'F:\new_sign_data\physical_v7'
    v8_path = r'F:\new_sign_data\physical_v8'
    v5 = os.listdir(v5_path)
    v7 = os.listdir(v7_path)
    v8 = os.listdir(v8_path)
    print(len(v5))
    print(len(v7))
    print(len(v8))
    union = list(set(v5).union(set(v7)))
    print(len(union))
    union = list(set(v8).union(set(union)))

    # for i in union:
    #     img_patch = os.path.join(r'F:\new_sign_data\physical_total' , i)
    #     img = cv2.imread(img_patch)
    #     cv2.imwrite(os.path.join(r'F:\new_sign_data\physical_union', i), img)
    # for i in os.listdir(r'F:\new_sign_data\physical2'):
    #     if i not in union:
    #         img_patch = os.path.join(r'F:\new_sign_data\physical_total', i)
    #         img = cv2.imread(img_patch)
    #         cv2.imwrite(os.path.join(r'F:\new_sign_data\physical_union1', i), img)


def rename():
    # 定义来源文件夹
    path_src = r'F:\new_test\small\dark'
    # 定义目标文件夹
    path_dst = r'F:\new_sign_data\test_digital'
    # 自定义格式，例如“报告-第X份”，第一个{}用于放序号，第二个{}用于放后缀
    rename_format = 'm_dark_{}'
    # 初始序号
    begin_num = 1

    def doc_rename(path_src, path_dst, begin_num):
        for i in os.listdir(path_src):
            # 获取原始文件名
            doc_src = os.path.join(path_src, i)
            # 重命名
            doc_name = rename_format.format(i)
            # 确定目标路径
            doc_dst = os.path.join(path_dst, doc_name)
            begin_num += 1
            os.rename(doc_src, doc_dst)

    # 运行函数
    doc_rename(path_src, path_dst, begin_num)


if __name__ == '__main__':
    # union_img()
    # sampled_batch = np.load('D:/PyCharmProject/DualAttentionAttack-main/src/textures/smile.npy')
    # print(sampled_batch.shape)
    # item = torch.IntTensor([1, 2, 4, 7, 3, 2])
    # value, indices = torch.topk(item, 5)
    # print("value:", value)
    # print("indices:", indices)
    # device = select_device('0', batch_size=1)
    # create_Dataset(device)
    # save_npz()
    # create_txt()
    # dec_img()
    # print(torch.hub.list('pytorch/vision'))
    # a = np.array([0])
    # print(a)
    # npz_png()
    # import math
    # print(math.degrees(math.atan(1/10)))
    # create_txt()
    # resize()
    # black_mm()
    # merge_mm()
    # crop_img()
    # save_npz()
    # gen_tex()
    # resize()
    # gen_white()
    # img_rotate()
    # resize()
    # bg_path = r'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages1\src\data2\phy_attack\res\black.jpg'
    # bg_img = Image.open(bg_path)
    # bg_img_arr = np.array(bg_img, dtype=np.uint8)
    # bg_img_arr[50:150 , 50:150] = 255
    # bg_img4 = Image.fromarray(bg_img_arr.astype(np.uint8)).convert('RGB')
    # bg_img4.save(
    #     r'D:\PyCharmProject\Full-coverage-camouflage-adversarial-attack-gh-pages1\src\data2\phy_attack\res\black1.jpg')
    # merge_mm()
    # gen_white()
    light()
    # rename()
    # create_Dataset()





