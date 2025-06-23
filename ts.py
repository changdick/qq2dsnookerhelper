# PocketRPG.exe  是进程名，能不能直接捕获窗口？
import win32gui
import win32ui
import win32con
import win32process
import psutil
from PIL import Image
import numpy as np
import time
import sys
import cv2
def find_window_by_process_name(process_name):
    """根据进程名查找窗口句柄"""
    hwnds = []
    
    def callback(hwnd, result):
        # 跳过不可见窗口
        if not win32gui.IsWindowVisible(hwnd):
            return True
            
        # 获取窗口所属进程ID
        _, found_pid = win32process.GetWindowThreadProcessId(hwnd)
        
        # 检查进程名
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['pid'] == found_pid and process_name.lower() in proc.info['name'].lower():
                    result.append(hwnd)
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
            
        return True
    
    # 枚举所有窗口
    win32gui.EnumWindows(callback, hwnds)
    print(len(hwnds))
    # test
    for hwnd in hwnds:

        left, top, right, bottom = win32gui.GetClientRect(hwnd)
    
        client_width, client_height = right - left, bottom - top
        
        # 获取窗口的整体大小和位置
        window_left, window_top, window_right, window_bottom = win32gui.GetWindowRect(hwnd)
        window_width = window_right - window_left
        window_height = window_bottom - window_top
        
        # 输出窗口大小信息
        print(f"GetClientRect 返回: left={left}, top={top}, right={right}, bottom={bottom}")
        
        print(f"窗口信息:")
        print(f"  客户区大小: {client_width}x{client_height} 像素")
        print(f"  窗口整体大小: {window_width}x{window_height} 像素")
        print(f"  窗口位置: 左上角({window_left},{window_top})")
        



    # 返回找到的第一个窗口，或None
    return hwnds[0] if hwnds else None

def capture_window(hwnd):
    """捕获窗口内容并返回图像"""
    # 获取窗口的客户区大小
    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    
    client_width, client_height = right - left, bottom - top
    
    # 获取窗口的整体大小和位置
    window_left, window_top, window_right, window_bottom = win32gui.GetWindowRect(hwnd)
    window_width = window_right - window_left
    window_height = window_bottom - window_top
    
    # 输出窗口大小信息
    # print(f"GetClientRect 返回: left={left}, top={top}, right={right}, bottom={bottom}")
    
    print(f"窗口信息:")
    print(f"  客户区大小: {client_width}x{client_height} 像素")
    print(f"  窗口整体大小: {window_width}x{window_height} 像素")
    print(f"  窗口位置: 左上角({window_left},{window_top})")
    
    if client_width <= 0 or client_height <= 0:
        print("警告: 窗口客户区大小为零，无法捕获图像")
        return None
    
    # 获取窗口DC
    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()
    
    # 创建位图对象
    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, client_width, client_height)
    save_dc.SelectObject(save_bitmap)
    
    # 复制屏幕内容到位图
    save_dc.BitBlt((0, 0), (client_width, client_height), mfc_dc, (0, 0), win32con.SRCCOPY)
    
    # 转换为numpy数组
    bmpinfo = save_bitmap.GetInfo()
    bmpstr = save_bitmap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype='uint8')
    img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)
    
    # 清理资源
    win32gui.DeleteObject(save_bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)
    
    # # 转换为RGB格式
    # img = img[:,:,0:3]
    # img = img[:,:,[2,1,0]] 
    # img = np.ascontiguousarray(img)

    img= cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return img

def main():
    # 从命令行参数获取进程名，默认为"PocketRPG.exe"
    process_name = sys.argv[1] if len(sys.argv) > 1 else "PocketRPG.exe"
    
    print(f"正在查找进程: {process_name}")
    
    # 查找窗口
    hwnd = find_window_by_process_name(process_name)
    
    if hwnd:
        # 获取窗口标题
        window_title = win32gui.GetWindowText(hwnd)
        print(f"找到窗口: {window_title}, 句柄: {hwnd}")
        
        # 捕获窗口内容
        img = capture_window(hwnd)
        
        if img is not None:
            # 转换为PIL图像并保存
            pil_image = Image.fromarray(img)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"window_capture_{timestamp}.png"
            pil_image.save(filename)
            print(f"窗口内容已保存到: {filename}")
        else:
            print("捕获窗口内容失败")
    else:
        print(f"未找到进程名为 '{process_name}' 的窗口")

if __name__ == "__main__":
    main()