import win32api, win32con

import win32gui
import win32ui
import win32con
import win32process
from time import sleep
import keyboard  # 键盘事件库
import threading
from PIL import ImageGrab, Image
import numpy as np
from time import time, sleep
from numpy.linalg import norm
# from table import Table
from tab1e import Table

import sys

import ts

is_running = True
is_manual_mode = False
manual_trigger = threading.Event()
virtual_pockets = []  # 虚拟袋口
def move(x, y):
    win32api.SetCursorPos([int(round(x)), int(round(y))])
    # pyautogui.moveTo(x, y)
    
def right(x, y):
    win32api.SetCursorPos([x, y])
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP | win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0, 0)

def left(x, y):
    win32api.SetCursorPos([x, y])
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP | win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

def getpos():
    return win32api.GetCursorPos() # win32api.GetCursorPos() 返回的坐标系 原点左上角，x轴向右，y轴向下

def grab():
    return ImageGrab.grab()

def snap(table, wnd_left=0, wnd_top=0 , call=print):
    if len(table.hitpts)==0: return
    x, y = getpos()
    pts = table.hitpts[:,:2] + table.loc + np.array([wnd_top, wnd_left])
    dif = norm(pts-(y, x), axis=-1)
    call(f"最近击球点距离: {dif.min():.2f}px")
    
    if dif.min()<8:
        n = np.argmin(dif)
        y, x = pts[n]
        obj = table.hitpts[n]
        move(x, y)
        # call('锁定目标，传递%d次，成功率%d%%'%(obj[3],obj[4]))
        call('锁定目标，传递%d次'%(obj[3]))


def analysis(img, tp='black8', goal=-1, maxiter=2, vpockets = []):
    """
    分析一个图片，获取原始图片后，直接调用该函数。
    该函数从图片中提取出桌面对象（调用 extract_table 并用 Table 的构造方法构造）。
    然后调用桌面对象的 solve 方法完成求解。
    参数:
        img: PIL.Image 或 numpy.ndarray
            要分析的原始图片。
        tp: str, 可选
            台球类型，默认为 'black8'。
        goal: int, 可选
            目标球类型，默认为 -1（通用）。
        maxiter: int, 可选
            传球次数，默认为 2。
    返回:
        tuple:
            (table, note)
            - table: Table 对象或 None，桌面对象，若分析失败则为 None。
            - note: str，分析结果说明或错误信息。
    """
    #img = Image.open('testimg/black8.png')
    from extract import extract_table
    table = extract_table(img, tp)
    if isinstance(table, str): 
        return (None, table)
    table = Table(*table, tp)
    vpockets = [(y - table.loc[0], x - table.loc[1] ) for x, y in vpockets]
    table.vpockets = vpockets.copy()  # 设置虚拟袋口
    # table.solve(goal, maxiter)
    table.solve_simple(goal)
    return (table, '共检测到%s条击球策略'%len(table.hitpts))

def hold(tp='snooker', goal=-1, maxiter=2, call=print):
      # 从命令行参数获取进程名，默认为"PocketRPG.exe"
    process_name = sys.argv[1] if len(sys.argv) > 1 else "PocketRPG.exe"
    
    print(f"正在查找进程: {process_name}")
    
    # 查找窗口
    hwnd = ts.find_window_by_process_name(process_name)
    if hwnd:
        # 获取窗口标题
        window_title = win32gui.GetWindowText(hwnd)
        print(f"找到窗口: {window_title}, 句柄: {hwnd}")
         # 新增：将窗口移动到(0, 0)位置
        # 参数说明：hwnd, x, y, width, height, repaint
        # 获取窗口当前大小
        # left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        # width, height = right - left, bottom - top
        # # 移动窗口到(0,0)并保持原大小
        # win32gui.MoveWindow(hwnd, 0, 0, width, height, True)
        # print(f"窗口已移动到(0,0)位置，大小: {width}x{height}")
    else:
        print(f"未找到进程 {process_name} 的窗口")
        return
    
    
    while True:
        sleep(0.5)
        # img = np.array(grab())[:,:,:3]
        img = ts.capture_window(hwnd)
        # 获取窗口当前位置
        left, top, _, _= win32gui.GetWindowRect(hwnd)
        table, note = analysis(img, tp, goal, maxiter)
        if table is None:
            call(note)
            continue
        else:
            call(note, call)
            snap(table, left, top)


def keyboard_listener():

    """监听键盘事件"""
    
    
    # 空格键：切换启动/暂停
    keyboard.add_hotkey('space', toggle_mode)
    keyboard.add_hotkey('esc', on_esc_press)
    # F10键：添加虚拟袋口
    keyboard.add_hotkey('f10', add_virtual_pocket)
    
    # F11键：清空虚拟袋口
    keyboard.add_hotkey('f11', clear_virtual_pockets)

    
    print("键盘监听已启动")
    keyboard.wait()  # 保持线程运行

def on_esc_press():
    print("ESC 键已按下！")
    setattr(sys.modules[__name__], "is_running", False)

def toggle_mode():
    """切换自动/手动模式"""
    global is_manual_mode
    is_manual_mode = not is_manual_mode
    mode = "手动模式" if is_manual_mode else "自动模式"
    print(f"已切换到{mode}")
    if not is_manual_mode:
        manual_trigger.set()

def add_virtual_pocket():
    global virtual_pockets
    x, y = win32api.GetCursorPos()
    virtual_pockets.append((x, y))
    print(f"已添加虚拟袋口: 屏幕坐标({x}, {y})")

def clear_virtual_pockets():
    """清除所有虚拟袋口"""
    global virtual_pockets
    virtual_pockets = []
    print("已清除所有虚拟袋口")


# 用户交互版的hold
def holdwithinteraction(tp='snooker', goal=-1, maxiter=2, call=print):
    global is_running, is_manual_mode, manual_trigger, virtual_pockets

    # 从命令行参数获取进程名，默认为"PocketRPG.exe"
    process_name = sys.argv[1] if len(sys.argv) > 1 else "PocketRPG.exe"
    
    print(f"正在查找进程: {process_name}")
    
    # 查找窗口
    hwnd = ts.find_window_by_process_name(process_name)
    if hwnd:
        # 获取窗口标题
        window_title = win32gui.GetWindowText(hwnd)
        print(f"找到窗口: {window_title}, 句柄: {hwnd}")
         # 新增：将窗口移动到(0, 0)位置
        # 参数说明：hwnd, x, y, width, height, repaint
        # 获取窗口当前大小
        # left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        # width, height = right - left, bottom - top
        # # 移动窗口到(0,0)并保持原大小
        # win32gui.MoveWindow(hwnd, 0, 0, width, height, True)
        # print(f"窗口已移动到(0,0)位置，大小: {width}x{height}")
    else:
        print(f"未找到进程 {process_name} 的窗口")
        return
    
    # 初始化键盘监听线程
    keyboard_thread = threading.Thread(target=keyboard_listener, daemon=True)
    keyboard_thread.start()

    
    
    while is_running:
        # print(is_running)
        if is_manual_mode:
            manual_trigger.wait()
            manual_trigger.clear()
            if not is_running:  # 检查是否在等待期间被终止
                break
        else:
            sleep(1)
            # img = np.array(grab())[:,:,:3]
            print("使用提示")
            print("   空格键-切换自动瞄准/手动游戏")
            print("   f10键--添加虚拟袋口")
            print("   f11键--清空虚拟袋口")
            print("   esc键--退出")
            img = ts.capture_window(hwnd)
            # 获取窗口当前位置
            left, top, _, _= win32gui.GetWindowRect(hwnd)
            vpockets = virtual_pockets.copy()
            vpockets = [(x - left, y - top) for x, y in vpockets]

            table, note = analysis(img, tp, goal, maxiter, vpockets=vpockets)
            if table is None:
                call(note)
                continue
            else:
                call(note, call)
                snap(table, left, top)

                
if __name__ == '__main__':
    holdwithinteraction()
