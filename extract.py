from PIL import Image
from skimage.io import imread
from skimage import color
from time import time
import numpy as np
from numpy.linalg import norm
import scipy.ndimage as ndimg
import matplotlib.pyplot as plt
import cv2

# RGB转换HSV空间
def rgb2hsv(rgb):
    hsv = np.zeros(rgb.shape, dtype=np.float32)
    cmax = rgb.max(axis=-1)
    #crng = rgb.ptp(axis=-1) # old
    crng = np.ptp(rgb, axis = -1)
    np.clip(cmax, 1, 255, out=hsv[:,:,1])
    np.divide(crng, hsv[:,:,1], out=hsv[:,:,1])
    np.divide(cmax, 255, out=hsv[:,:,2])
    maxidx = np.argmax(rgb, axis=-1).ravel()
    colrgb = rgb.reshape(-1,3)
    idx = np.arange(colrgb.shape[0])
    lut = np.array([[1,2,0],[2,0,1]], dtype=np.uint8)
    h = (colrgb[idx, lut[0][maxidx]]).astype(np.float32)
    h -= colrgb[idx, lut[1][maxidx]]
    h[h==0] = 1
    np.clip(crng, 1, 255, out=crng)
    h /= crng.ravel()
    h += np.array([0,2,4], dtype=np.uint8)[maxidx]
    h /= 6; h %= 1
    hsv[:,:,0] = h.reshape(hsv.shape[:2])
    return hsv

# 制作HSV索引表
def make_lut():
    arr = np.mgrid[0:256,0:256,0:256].reshape(3,-1).T
    arr = arr.astype(np.uint8)
    lut = rgb2hsv(arr.reshape(1,-1,3))
    lut = (lut[0,:,0]*255).astype(np.uint8)
    return lut.reshape(256,256,256)

# 利用索引进行RGB到HSV转换
def rgb2hsv_lut(rgb, lut=[None]):
    if lut[0] is None: lut[0] = make_lut()
    r,g,b = rgb.reshape(-1,3).T
    return lut[0][r,g,b].reshape(rgb.shape[:2])
    
# 计算角度
def angleX(v):
    a = np.arccos(v[:,0] / (norm(v[:,:2], axis=1)+1e-5))
    return np.where(v[:,1]>=0,a ,np.pi * 2 - a)

# 精确定位, 根据圆心和采样点，组建法方程，进行最小二乘估计
def exactly(O, r, pts):
    n = len(pts)
    B = np.zeros((n*2, n+3))
    L = np.zeros(n*2)
    appro = np.zeros(n+3)
    appro[:n] = angleX(pts-O)
    appro[n:] = [O[0], O[1], r]
    try:
        for i in range(2): # 两次迭代，确保达到稳定
            L[::2] = appro[n]+appro[-1]*np.cos(appro[:n])-pts[:,0]
            L[1::2] = appro[n+1]+appro[-1]*np.sin(appro[:n])-pts[:,1]
            B[range(0,n*2,2),range(n)] = -appro[-1]*np.sin(appro[:n])
            B[range(1,n*2,2),range(n)] = appro[-1]*np.cos(appro[:n])
            B[::2,n],B[1::2,n+1] = 1, 1
            B[::2,-1] = np.cos(appro[:n])
            B[1::2,-1] = np.sin(appro[:n])
            NN = np.linalg.inv(np.dot(B.T,B))
            x = np.dot(NN, np.dot(B.T,L))
            v = np.dot(B,x)-L
            appro -= x
    except:
        print(O, r, pts)
    if not(appro[-1]>5 and appro[-1]<50): 
        return (None, None), None
    return appro[[-3,-2]], appro[-1]

#a = np.arccos(v[:,0] / norm(v[:,:2], axis=1))
# 查找背景
def find_ground(img, tor=5):
    """
    注意：此函数输入的img是经rgb2hsv转换，只保留了h通道的图像。 只有2个维度，相当于是二维数组。

    """
    r, c = np.array(img.shape[:2])//2    # r和c是一个数，（r，c）就是坐标值
    center = img[r-100:r+100, c-100:c+100]    # 以r和c为中心，上下左右各阔开100的区域，在这个区域中找最多的色调值，这是有效的。 经检查，绿色桌面，色调值就是80，不论哪个分辨率
    back = np.argmax(np.bincount(center.ravel()))  # center中最多的色调值，斯诺克绿桌就是80
    msk = np.abs(img.astype(np.int16) - back)<tor
    lab, n = ndimg.label(msk)
    hist = np.bincount(lab.ravel())
    if hist[1:].max() < 1e4: return None   # label提取连通区域，为了确认足够大，从而确认是背景
    if np.argmax(hist[1:])==0: return None
    msk = lab == np.argmax(hist[1:]) + 1
    sr, sc = ndimg.find_objects(msk)[0]    # 返回两个切片对象， sr是行上的切片对象，sc是列上的切片对象
    loc = sr.start, sc.start                 # 以桌面左上角坐标作为loc ，二元组，左上角坐标
    size = sr.stop - loc[0], sc.stop - loc[1]  # 桌面的大小size， size作为一个元组，接受两个值， 二元组，行长度，列长度
    
    return loc, size, sr, sc, msk[sr, sc]




def find_ground_opencv(img, tor=5):
    # img: HSV图像的H通道，二维数组
    r, c = np.array(img.shape[:2]) // 2
    center = img[r-100:r+100, c-100:c+100]
    back = np.argmax(np.bincount(center.ravel()))
    
    # 背景色掩码
    msk = (np.abs(img.astype(np.int16) - back) < tor).astype(np.uint8)

    # 连通域标记（返回的 labels 是每个像素的连通域编号）
    num_labels, labels = cv2.connectedComponents(msk)

    # 每个 label 的像素个数
    hist = np.bincount(labels.ravel())

    if len(hist) <= 1 or hist[1:].max() < 1e4:
        return None
    
    max_label = 1 + np.argmax(hist[1:])
    msk_max = (labels == max_label).astype(np.uint8)

    # 找到最大连通区域的边界框
    contours, _ = cv2.findContours(msk_max, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    x, y, w, h = cv2.boundingRect(contours[0])

    loc = (y, x)  # 注意顺序：行（y）、列（x）
    size = (h, w)
    sr = slice(y, y + h)
    sc = slice(x, x + w)

    return loc, size, sr, sc, msk_max[y:y + h, x:x + w]

# 查找一个球
def find_one(img, cs, r=16, a=30):
    # 输入的img：二维数组，背景为true，球为false
    h, w = img.shape
    if cs[0]<r+1 or cs[1]<r+1 or cs[0]>h-r-1 or cs[1]>w-r-1:  # 初始的r是个预设的搜索范围。
        return (None, None), None
    rs, pts = np.arange(r), []
    for i in np.linspace(0, np.pi*2, a, endpoint=False):
        rcs = rs[:,None] * (np.cos(i), np.sin(i)) + cs
        rcs = rcs.round().astype(int).T
        ns = rs[img[rcs[0], rcs[1]]]  # 布尔数组作为索引，返回一个一维数组。ns中最小的值是这个方向最接近圆心的背景的点到圆心的距离
        if len(ns)==0: continue
        pts.append(rcs.T[ns.min()])    # ns.min()就是该方向（i）上最接近圆心的背景点到圆心的径向距离，作为rcs的索引，提取出这个方向上的点坐标，append到pts中
    if len(pts)<10: return (None, None), None
    return exactly(cs, r, np.array(pts))

# 检测球
def find_ball(img):
    dist = ndimg.binary_dilation(img, np.ones((13, 13)))
    dist[:,[0,-1]] = 0; dist[[0,-1],:] = 0
    lab, n = ndimg.label(~dist)
    objs = ndimg.find_objects(lab)[1:]
    cs = [(i.start+i.stop, j.start+j.stop) for i,j in objs]
    balls = []
    for i in np.array(cs)/2:
        (r, c), ra = find_one(img, i)
        if not ra is None: balls.append([r, c, ra])
    if len(balls)==0: return balls
    balls = np.array(balls)
    balls[:,2] = balls[:,2].mean()-0.5
    return balls

def find_ball_opencv(img):
    # img: 背景掩码（背景为True/1，球为False/0）
    
    # OpenCV要求uint8类型，背景为255，球为0
    mask = (img == 1).astype(np.uint8) * 255
    
    # 膨胀操作，替代 binary_dilation
    kernel = np.ones((13, 13), np.uint8)
    dist = cv2.dilate(mask, kernel)

    # 边界清除
    dist[0, :] = 0
    dist[-1, :] = 0
    dist[:, 0] = 0
    dist[:, -1] = 0

    # 求非背景区域（原始 img 中球的位置）
    inv = cv2.bitwise_not(dist)

    # 连通区域标记
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv)

    # 忽略背景标签0
    balls = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        cx, cy = centroids[i]
        if w < 5 or h < 5 or area > 120:  # 可选：排除过小区域
            continue

        # 尝试拟合圆（替代 find_one）
        (r, c), ra = find_one(img, (cy, cx))  # 你已有的精细定位逻辑
        if ra is not None:
            balls.append([r, c, ra])
    
    if len(balls) == 0:
        return []

    balls = np.array(balls)
    balls[:, 2] = balls[:, 2].mean() - 0.5
    return balls


# 提取颜色
def detect_color(img, balls, mode='snooker'):
    r = int(balls[0,2]) - 1
    rcs = np.mgrid[-r:r+1, -r:r+1].reshape(2,-1).T
    rcs = rcs[norm(rcs, axis=1) < r]   # 这两句构造rcs，是一个相对坐标列表，是用来和圆心相加生成rs和cs的
    colors = []
    for r,c in balls[:,:2]:
        rs, cs = (rcs + (int(r), int(c))).T
        colors.append(img[rs, cs])
    colors = np.array(colors).astype(np.int16)
    colors = np.sort(colors, axis=1)
    colors = colors[:,len(rcs)//4:-len(rcs)//4]
    if mode=='snooker':
        snklut = [21, 0, 34, 73, 12, 171, 221, 42]
        cs = [np.argmax(np.bincount(i)) for i in colors]
        diff = np.abs(np.array(cs)[:,None] - snklut)
        return np.argmin(diff, axis=-1)
    
    if mode=='black8':
        bins = np.array([np.bincount(i, minlength=256) for i in colors])
        mean = np.argmax(bins, axis=-1)
        std = (np.std(colors, axis=1)>1) + 1
        std[(std==1) & (np.abs(mean-42)<3)] = 7
        n = (np.abs(colors-28)<3).sum(axis=1)
        n = bins[:,25:30].max(axis=1)
        #print(mean)
        #print(np.bincount(colors[5]))
        #print(np.bincount(colors[9]))
        std[np.argmax(n)] = 0
        return std

# lut = np.load('lut.npy')
# 提取球桌信息
def extract_table(img, mode='snooker'):
    """
    提取球桌信息：先做色彩空间转换，调用find_ground找背景，调用find_ball找球，再用detect_color识别球的种类。
    """
    #hsv = (rgb2hsv(img[:,:,:3]) * 255).astype(np.uint8)
    hsv = rgb2hsv_lut(img)
    ground = find_ground_opencv(hsv)
    if ground is None: return '未检测到球桌，请勿遮挡'
    loc, size, sr, sc, back = ground    # find_ground提取出来的五元组 就长这样，back是表示桌面的二维数组，每个元素是这个像素是桌面或不是
    
    
    r, c = np.array(hsv.shape[:2])//2    # r和c是一个数，（r，c）就是坐标值
    center = hsv[r-100:r+100, c-100:c+100]    # 以r和c为中心，上下左右各阔开100的区域，在这个区域中找最多的色调值，这是有效的。 经检查，绿色桌面，色调值就是80，不论哪个分辨率
    bgcolor = np.argmax(np.bincount(center.ravel()))  # center中最多的色调值，斯诺克绿桌就是80
    msk = np.abs(hsv.astype(np.int16) - bgcolor)<5   


    balls = find_ball_opencv(msk[sr, sc])  # 在背景中找球，返回一个二维数组，每行是一个球的坐标和半径
    if len(balls)==0: return '全部球已入袋'
    tps = detect_color(hsv[sr, sc], balls, mode)
    balls = np.hstack((balls, tps[:,None]))
    return loc, size, img[sr, sc], balls
    
if __name__ == '__main__':
    img = imread('window_capture_20250614_194521.png')[:,:,:3]
    start = time()
    #hsv = (rgb2hsv(img[:,:,:0]) * 255).astype(np.uint8)
    ax = plt.subplot(221)
    ax.imshow(img)

    hsv = rgb2hsv_lut(img)
    print('to hsv', time()-start)
    ax = plt.subplot(222)
    ax.imshow(hsv)

    start = time()
    loc, size, sr, sc, back = find_ground(hsv)
    print('back', time()-start)
    ax = plt.subplot(223)
    ax.imshow(back)

    start = time()
    balls = find_ball(back)

    ax = plt.subplot(224)
    ax.imshow(img[sr, sc])
    ax.plot(balls[:,1], balls[:,0], 'r.')

    plt.show()

    print('ball', time()-start)

    start = time()
    tps = detect_color(hsv[sr, sc], balls)
    
    print('detect', time()-start)
    print(tps)