from tkinter import *
import random
import numpy as np
import math

def dispatch_color(edge_color , E):
    for egde_id in range(len(E)):
        color = '#' + str("%03d" % random.randint(0, 255))[2:] + str("%03d" % random.randint(0, 255))[2:] + str("%03d" %random.randint(0, 255))[2:]
        edge_color.append(color)
    return edge_color

def get_info(U, MAX_EP_STEPS):
    x_min, x_Max, y_min, y_Max = np.inf, -np.inf, np.inf, -np.inf
    # x axis
    for user in U:
       if(max(user.mob[:, 0]) > x_Max):
           x_Max = max(user.mob[:, 0])
       if(min(user.mob[:, 0]) < x_min):
           x_min = min(user.mob[:, 0])
    # y axis
    for user in U:
        if (max(user.mob[:MAX_EP_STEPS, 1]) > y_Max):
            y_Max = max(user.mob[:MAX_EP_STEPS, 1])
        if (min(user.mob[:MAX_EP_STEPS, 1]) < y_min):
            y_min = min(user.mob[:MAX_EP_STEPS, 1])
    return x_min, x_Max, y_min, y_Max

#####################  hyper parameters  ####################
MAX_SCREEN_SIZE = 1000
EDGE_SIZE = 20
USER_SIZE = 10

#####################  User  ####################
class oval_User:
    def __init__(self, canvas, color, user_id):
        self.user_id = user_id
        self.canvas = canvas
        self.id = canvas.create_oval(500, 500, 500 + USER_SIZE, 500 + USER_SIZE, fill=color)

    def draw(self, vector, edge_color, user):
        info = self.canvas.coords(self.id)
        self.canvas.delete(self.id)
        # connect
        if user.req.state != 5 and user.req.state != 6:
            self.id = self.canvas.create_oval(info[0], info[1], info[2], info[3], fill=edge_color)
        # not connected
        else:
            # disconnection
            if user.req.state == 5:
                self.id = self.canvas.create_oval(info[0], info[1], info[2], info[3], fill="red")
            # migration
            elif user.req.state == 6:
                self.id = self.canvas.create_oval(info[0], info[1], info[2], info[3], fill="green")
        # move the user
        self.canvas.move(self.id, vector[0][0], vector[0][1])

#####################  Edge  ####################
class oval_Edge:
    def __init__(self, canvas, color, edge_id):
        self.edge_id = edge_id
        self.canvas = canvas
        self.id = canvas.create_oval(500, 500, 500 + EDGE_SIZE, 500 + EDGE_SIZE, fill=color)

    def draw(self, vector):
        self.canvas.move(self.id, vector[0][0], vector[0][1])

#####################  convas  ####################
class Demo:
    def __init__(self, E, U, O, MAX_EP_STEPS):
        # create canvas
        self.x_min, self.x_Max, self.y_min, self.y_Max = get_info(U, MAX_EP_STEPS)
        self.tk = Tk()
        self.tk.title("Simulation: Resource Allocation in Egde Computing Environment")
        self.tk.resizable(0, 0)
        self.tk.wm_attributes("-topmost", 1)
        self.canvas = Canvas(self.tk, width=MAX_SCREEN_SIZE, height=1000, bd=0, highlightthickness=0, bg='black')
        self.canvas.pack()
        self.tk.update()
        x_range = self.x_Max - self.x_min
        y_range = self.y_Max - self.y_min
        self.rate = x_range/y_range
        if self.rate > 1:
            self.x_rate = (MAX_SCREEN_SIZE / x_range)
            self.y_rate = (MAX_SCREEN_SIZE / y_range) * (1/self.rate)
        else:
            self.x_rate = (MAX_SCREEN_SIZE / x_range) * (self.rate)
            self.y_rate = (MAX_SCREEN_SIZE / y_range)

        self.edge_color = []
        self.edge_color = dispatch_color(self.edge_color, E)
        self.oval_U, self.oval_E = [], []
        # initialize the object
        for edge_id in range(len(E)):
            self.oval_E.append(oval_Edge(self.canvas, self.edge_color[edge_id], edge_id))
        for user_id in range(len(U)):
            self.oval_U.append(oval_User(self.canvas, self.edge_color[int(O[user_id])], user_id))

    def draw(self, E, U, O):
        # edge
        edge_vector = np.zeros((1, 2))
        for edge in E:
            edge_vector[0][0] = (edge.loc[0] - self.x_min) * self.x_rate - self.canvas.coords(self.oval_E[edge.edge_id].id)[0]
            edge_vector[0][1] = (edge.loc[1] - self.y_min) * self.y_rate - self.canvas.coords(self.oval_E[edge.edge_id].id)[1]
            self.oval_E[edge.edge_id].draw(edge_vector)
        # user
        user_vector = np.zeros((1, 2))
        for user in U:
            user_vector[0][0] = (user.loc[0][0] - self.x_min) * self.x_rate - self.canvas.coords(self.oval_U[user.user_id].id)[0]
            user_vector[0][1] = (user.loc[0][1] - self.y_min) * self.y_rate - self.canvas.coords(self.oval_U[user.user_id].id)[1]
            self.oval_U[user.user_id].draw(user_vector, self.edge_color[int(O[user.user_id])], user)
        # 快速刷新屏幕
        self.tk.update_idletasks()
        self.tk.update()

#####################  Outer parameter  ####################
class UE():
    def __init__(self, user_id, data_num):
        self.user_id = user_id  # number of the user
        self.loc = np.zeros((1, 2))
        self.num_step = 0  # the number of step

        # calculate num_step and define self.mob
        data_num = str("%03d" % (data_num + 1))  # plus zero
        file_name = LOCATION + "_30sec_" + data_num + ".txt"
        file_path = LOCATION + "/" + file_name
        f = open(file_path, "r")
        f1 = f.readlines()
        data = 0
        for line in f1:
            data += 1
        self.num_step = data * 30
        self.mob = np.zeros((self.num_step, 2))

        # write data to self.mob
        now_sec = 0
        for line in f1:
            for sec in range(30):
                self.mob[now_sec + sec][0] = line.split()[1]  # x
                self.mob[now_sec + sec][1] = line.split()[2]  # y
            now_sec += 30

    def mobility_update(self, time):  # t: second
        if time < len(self.mob[:, 0]):
            self.loc[0] = self.mob[time]   # x

        else:
            self.loc[0][0] = np.inf
            self.loc[0][1] = np.inf

class EdgeServer():
    def __init__(self, edge_id, loc):
        self.edge_id = edge_id  # edge server number
        self.loc = loc


