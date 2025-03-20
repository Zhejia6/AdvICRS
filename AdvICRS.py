import os
import glob
import numpy as np
import cv2
from scipy.optimize import differential_evolution
from mmdet.apis import init_detector, inference_detector
import numpy as np
import random


# 初始化YOLOv3模型
config_car_yolov3 = '/root/autodl-tmp/code/configs/configs/yolov3.py'
checkpoint_car_inf_yolov3 = '/root/autodl-tmp/code/weights/my_custom_dataset/yolov3_909.pth'
model_car_inf_yolov3 = init_detector(config_car_yolov3, checkpoint_car_inf_yolov3, device='cuda:0')

# # 初始化DETR模型
# config_car_detr = '/root/autodl-tmp/code/configs/configs/detr.py'
# checkpoint_car_inf_detr = '/root/autodl-tmp/code/weights/my_custom_dataset/detr_912.pth'
# model_car_inf_detr = init_detector(config_car_detr, checkpoint_car_inf_detr, device='cuda:0')

# # 初始化 Mask R-CNN 模型
# config_car_mask = '/root/autodl-tmp/code/configs/configs/mask.py'
# checkpoint_car_inf_mask = '/root/autodl-tmp/code/weights/my_custom_dataset/mask_895.pth'
# model_car_inf_mask = init_detector(config_car_mask, checkpoint_car_inf_mask, device='cuda:0')

# # 初始化 faster模型
# config_car_faster = '/root/autodl-tmp/code/configs/configs/faster.py'
# checkpoint_car_inf_faster = '/root/autodl-tmp/code/weights/my_custom_dataset/faster_908.pth'
# model_car_inf_faster = init_detector(config_car_faster, checkpoint_car_inf_faster, device='cuda:0')

# #初始化libra模型
# config_car_libra = '/root/autodl-tmp/code/configs/configs/libra.py'
# checkpoint_car_inf_libra = '/root/autodl-tmp/code/weights/my_custom_dataset/libra_880.pth'
# model_car_inf_libra = init_detector(config_car_libra, checkpoint_car_inf_libra, device='cuda:0')

# #初始化retina模型
# config_car_retina = '/root/autodl-tmp/code/configs/configs/retina.py'
# checkpoint_car_inf_retina = '/root/autodl-tmp/code/weights/my_custom_dataset/retina_930.pth'
# model_car_inf_retina = init_detector(config_car_retina, checkpoint_car_inf_retina, device='cuda:0')

# #初始化yolof模型
# config_car_yolof = '/root/autodl-tmp/code/configs/configs/yolof.py'
# checkpoint_car_inf_yolof = '/root/autodl-tmp/code/weights/my_custom_dataset/yolof_921.pth'
# model_car_inf_yolof = init_detector(config_car_yolof, checkpoint_car_inf_yolof, device='cuda:0')

# #初始化yolox模型
# config_car_yolox = '/root/autodl-tmp/code/configs/configs/yolox.py'
# checkpoint_car_inf_yolox = '/root/autodl-tmp/code/weights/my_custom_dataset/yolox_893.pth'
# model_car_inf_yolox = init_detector(config_car_yolox, checkpoint_car_inf_yolox, device='cuda:0')

# #初始化deformable_detr模型
# config_car_deformable_detr = '/root/autodl-tmp/code/configs/configs/deformable_detr.py'
# checkpoint_car_inf_deformable_detr = '/root/autodl-tmp/code/weights/my_custom_dataset/deformable_detr_928.pth'
# model_car_inf_deformable_detr = init_detector(config_car_deformable_detr, checkpoint_car_inf_deformable_detr, device='cuda:0')




# YOLOv3 检测封装函数
def yolov3_inf(img_path, model):
    img = cv2.imread(img_path)
    results = inference_detector(model, img)
    bboxes = [bbox for result in results for bbox in result if bbox[4] > 0.5]  # 筛选置信度 > 0.5 的检测框
    return bboxes
# YOLOv3检测函数
def detection(img, model):
    results = inference_detector(model, img)
    bboxes = []
    confidences = []
    for result in results:
        for bbox in result:
            x1, y1, x2, y2, score = bbox
            if score > 0.5:
                bboxes.append((x1, y1, x2, y2))
                confidences.append(score)
    return bboxes, confidences

# # DETR 检测封装函数
# def detr_inf(img_path, model):
#     img = cv2.imread(img_path)
#     results = inference_detector(model, img)
#     bboxes = [bbox for result in results for bbox in result if bbox[4] > 0.5]  
#     return bboxes
# # DETR 检测函数
# def detection(img, model):
#     results = inference_detector(model, img)
#     bboxes = []
#     confidences = []
#     for result in results:
#         for bbox in result:
#             x1, y1, x2, y2, score = bbox
#             if score > 0.5:
#                 bboxes.append((x1, y1, x2, y2))
#                 confidences.append(score)
#     return bboxes, confidences


# # mask 检测封装函数
# def mask_inf(img_path, model):
#     img = cv2.imread(img_path)
#     results = inference_detector(model, img)
    
#    
#     if len(results) == 2: 
#         results = results[0] 

#    
#     bboxes = [bbox for result in results for bbox in result if bbox[4] > 0.5]
#     return bboxes
# # mask 检测函数
# def detection(img, model):
#     results = inference_detector(model, img)
#     bboxes = []
#     confidences = []

#    
#     if len(results) == 2: 
#         results = results[0]  

#     
#     for result in results:
#         for bbox in result:
#             x1, y1, x2, y2, score = bbox
#             if score > 0.5:  
#                 bboxes.append((x1, y1, x2, y2))
#                 confidences.append(score)
    
#     return bboxes, confidences


# # faster检测封装函数
# def faster_inf(img_path, model):
#     img = cv2.imread(img_path)
#     results = inference_detector(model, img)
    
#     
#     if len(results) == 2:  
#         results = results[0]  
#    
#     bboxes = [bbox for result in results for bbox in result if bbox[4] > 0.5]
#     return bboxes
# # faster 检测函数
# def detection(img, model):
#     results = inference_detector(model, img)
#     bboxes = []
#     confidences = []

#     
#     if len(results) == 2: 
#         results = results[0]  
#     # 遍历边界框结果
#     for result in results:
#         for bbox in result:
#             x1, y1, x2, y2, score = bbox
#             if score > 0.5:  
#                 bboxes.append((x1, y1, x2, y2))
#                 confidences.append(score)
    
#     return bboxes, confidences


# #libra 检测封装函数
# def libra_inf(img_path, model):
#     img = cv2.imread(img_path)
#     results = inference_detector(model, img)
#     bboxes = [bbox for result in results for bbox in result if bbox[4] > 0.5] 
#     return bboxes
# # libra检测函数
# def detection(img, model):
#     results = inference_detector(model, img)
#     bboxes = []
#     confidences = []
#     for result in results:
#         for bbox in result:
#             x1, y1, x2, y2, score = bbox
#             if score > 0.5:
#                 bboxes.append((x1, y1, x2, y2))
#                 confidences.append(score)
#     return bboxes, confidences

# #retina 检测封装函数
# def retina_inf(img_path, model):
#     img = cv2.imread(img_path)
#     results = inference_detector(model, img)
#     bboxes = [bbox for result in results for bbox in result if bbox[4] > 0.5] 
#     return bboxes
# # retina检测函数
# def detection(img, model):
#     results = inference_detector(model, img)
#     bboxes = []
#     confidences = []
#     for result in results:
#         for bbox in result:
#             x1, y1, x2, y2, score = bbox
#             if score > 0.5:
#                 bboxes.append((x1, y1, x2, y2))
#                 confidences.append(score)
#     return bboxes, confidences


# #yolof 检测封装函数
# def yolof_inf(img_path, model):
#     img = cv2.imread(img_path)
#     results = inference_detector(model, img)
#     bboxes = [bbox for result in results for bbox in result if bbox[4] > 0.5] 
#     return bboxes
# # yolof检测函数
# def detection(img, model):
#     results = inference_detector(model, img)
#     bboxes = []
#     confidences = []
#     for result in results:
#         for bbox in result:
#             x1, y1, x2, y2, score = bbox
#             if score > 0.5:
#                 bboxes.append((x1, y1, x2, y2))
#                 confidences.append(score)
#     return bboxes, confidences

# #yolox 检测封装函数
# def yolox_inf(img_path, model):
#     img = cv2.imread(img_path)
#     results = inference_detector(model, img)
#     bboxes = [bbox for result in results for bbox in result if bbox[4] > 0.5]  
#     return bboxes
# # yolox检测函数
# def detection(img, model):
#     results = inference_detector(model, img)
#     bboxes = []
#     confidences = []
#     for result in results:
#         for bbox in result:
#             x1, y1, x2, y2, score = bbox
#             if score > 0.5:
#                 bboxes.append((x1, y1, x2, y2))
#                 confidences.append(score)
#     return bboxes, confidences

# #deformable_detr 检测封装函数
# def deformable_detr_inf(img_path, model):
#     img = cv2.imread(img_path)
#     results = inference_detector(model, img)
#     bboxes = [bbox for result in results for bbox in result if bbox[4] > 0.5] 
#     return bboxes
# # deformable_detr检测函数
# def detection(img, model):
#     results = inference_detector(model, img)
#     bboxes = []
#     confidences = []
#     for result in results:
#         for bbox in result:
#             x1, y1, x2, y2, score = bbox
#             if score > 0.5:
#                 bboxes.append((x1, y1, x2, y2))
#                 confidences.append(score)
#     return bboxes, confidences


def generate_control_points_in_upper_bbox(X1, Y1, X2, Y2):
    width_start = X1 + (X2 - X1) / 5  
    width_end = X1 + 4 * (X2 - X1) / 5  
    width_step = (width_end - width_start) / 3 

    height_start = Y1 + (Y2 - Y1) / 5  
    height_end = Y1 + 7 * (Y2 - Y1) / 10 
    height_step = (height_end - height_start) / 3  

    P1 = [width_start + np.random.uniform(0, width_step), height_start + np.random.uniform(0, height_step)]
    P2 = [width_start + np.random.uniform(width_step, 2 * width_step), height_start + np.random.uniform(0, height_step)]
    P3 = [width_start + np.random.uniform(2 * width_step, 3 * width_step), height_start + np.random.uniform(0, height_step)]
    P4 = [width_start + np.random.uniform(2 * width_step, 3 * width_step), height_start + np.random.uniform(height_step, 2 * height_step)]
    P5 = [width_start + np.random.uniform(2 * width_step, 3 * width_step), height_start + np.random.uniform(2 * height_step, 3 * height_step)]
    P6 = [width_start + np.random.uniform(width_step, 2 * width_step), height_start + np.random.uniform(2 * height_step, 3 * height_step)]
    P7 = [width_start + np.random.uniform(0, width_step), height_start + np.random.uniform(2 * height_step, 3 * height_step)]
    P8 = [width_start + np.random.uniform(0, width_step), height_start + np.random.uniform(height_step, 2 * height_step)]

    return [P1, P2, P3, P4, P5, P6, P7, P8]




def clip_catmull_rom_points(control_points, X1, Y1, X2, Y2):
    width_start = X1 + (X2 - X1) / 5
    width_end = X1 + 4 * (X2 - X1) / 5
    width_step = (width_end - width_start) / 3

    height_start = Y1 + (Y2 - Y1) / 5
    height_end = Y1 + 7 * (Y2 - Y1) / 10
    height_step = (height_end - height_start) / 3

    
    if not (width_start <= control_points[0][0] <= width_start + width_step):
        control_points[0][0] = random.uniform(width_start, width_start + width_step)
    if not (height_start <= control_points[0][1] <= height_start + height_step):
        control_points[0][1] = random.uniform(height_start, height_start + height_step)

   
    if not (width_start + width_step <= control_points[1][0] <= width_start + 2 * width_step):
        control_points[1][0] = random.uniform(width_start + width_step, width_start + 2 * width_step)
    if not (height_start <= control_points[1][1] <= height_start + height_step):
        control_points[1][1] = random.uniform(height_start, height_start + height_step)

    
    if not (width_start + 2 * width_step <= control_points[2][0] <= width_end):
        control_points[2][0] = random.uniform(width_start + 2 * width_step, width_end)
    if not (height_start <= control_points[2][1] <= height_start + height_step):
        control_points[2][1] = random.uniform(height_start, height_start + height_step)

   
    if not (width_start + 2 * width_step <= control_points[3][0] <= width_end):
        control_points[3][0] = random.uniform(width_start + 2 * width_step, width_end)
    if not (height_start + height_step <= control_points[3][1] <= height_start + 2 * height_step):
        control_points[3][1] = random.uniform(height_start + height_step, height_start + 2 * height_step)

    
    if not (width_start + 2 * width_step <= control_points[4][0] <= width_end):
        control_points[4][0] = random.uniform(width_start + 2 * width_step, width_end)
    if not (height_start + 2 * height_step <= control_points[4][1] <= height_end):
        control_points[4][1] = random.uniform(height_start + 2 * height_step, height_end)

    
    if not (width_start + width_step <= control_points[5][0] <= width_start + 2 * width_step):
        control_points[5][0] = random.uniform(width_start + width_step, width_start + 2 * width_step)
    if not (height_start + 2 * height_step <= control_points[5][1] <= height_end):
        control_points[5][1] = random.uniform(height_start + 2 * height_step, height_end)

    
    if not (width_start <= control_points[6][0] <= width_start + width_step):
        control_points[6][0] = random.uniform(width_start, width_start + width_step)
    if not (height_start + 2 * height_step <= control_points[6][1] <= height_end):
        control_points[6][1] = random.uniform(height_start + 2 * height_step, height_end)

    
    if not (width_start <= control_points[7][0] <= width_start + width_step):
        control_points[7][0] = random.uniform(width_start, width_start + width_step)
    if not (height_start + height_step <= control_points[7][1] <= height_start + 2 * height_step):
        control_points[7][1] = random.uniform(height_start + height_step, height_start + 2 * height_step)

    return control_points

# 生成 Catmull-Rom 曲线
def catmull_rom_spline_with_tau(P0, P1, P2, P3, tau=0.5, num_points=100):
    def tj(ti, Pi, Pj):
        return ti + (np.linalg.norm(Pj - Pi) ** tau)

    t0 = 0
    t1 = tj(t0, P0, P1)
    t2 = tj(t1, P1, P2)
    t3 = tj(t2, P2, P3)

    t = np.linspace(t1, t2, num_points)

    A1 = (t1 - t)[:, None] / (t1 - t0) * P0 + (t - t0)[:, None] / (t1 - t0) * P1
    A2 = (t2 - t)[:, None] / (t2 - t1) * P1 + (t - t1)[:, None] / (t2 - t1) * P2
    A3 = (t3 - t)[:, None] / (t3 - t2) * P2 + (t - t2)[:, None] / (t3 - t2) * P3

    B1 = (t2 - t)[:, None] / (t2 - t0) * A1 + (t - t0)[:, None] / (t2 - t0) * A2
    B2 = (t3 - t)[:, None] / (t3 - t1) * A2 + (t - t1)[:, None] / (t3 - t1) * A3

    C = (t2 - t)[:, None] / (t2 - t1) * B1 + (t - t1)[:, None] / (t2 - t1) * B2

    return C

def add_catmull_rom_curve(img, valid_bboxes):
    for bbox in valid_bboxes:
        X1, Y1, X2, Y2 = map(int, bbox[:4])  
        bbox_width = X2 - X1  
        bbox_height = Y2 - Y1  

        
        R = max(1, int(bbox_height / 200))  

        control_points = generate_control_points_in_upper_bbox(X1, Y1, X2, Y2)

        all_curve_points = []
        for i in range(8):
            P0, P1, P2, P3 = control_points[i % 8], control_points[(i + 1) % 8], control_points[(i + 2) % 8], control_points[(i + 3) % 8]
            curve_points = catmull_rom_spline_with_tau(np.array(P0), np.array(P1), np.array(P2), np.array(P3), tau=0.5, num_points=100)
            all_curve_points.append(curve_points)

        all_curve_points = np.vstack(all_curve_points)

        for point in all_curve_points:
            if X1 <= point[0] <= X2 and Y1 <= point[1] <= Y2:
                cv2.circle(img, (int(point[0]), int(point[1])), R, (0, 0, 0), -1)

    return img





# # 攻击统计
# ASR = 0      
# Query = 0    
# count_all = 0  

# 进化策略参数
sigma0 = 0.11   
gamma = 0.05*sigma0    
seed = 50       
step = 15       

# 初始化种群
def initialize_population(X1, Y1, X2, Y2, num_particles=seed):
    population = np.zeros((num_particles, 8, 2))
    for i in range(num_particles):
        population[i] = generate_control_points_in_upper_bbox(X1, Y1, X2, Y2)
    return population

# 适应度函数
def fitness_function(conf):
    return 1 / (conf + 1e-6)

# 变异操作
def mutation(individual, current_step, sigma0=sigma0, gamma=gamma):
    
    sigma = sigma0 * np.exp(-gamma * current_step)
   
    omega = np.random.uniform(0, 1.0)
    
    noise = np.random.normal(0, sigma, size=individual.shape)
    
    mutated_individual = omega * individual + (1 - omega) * (individual + noise)
    return mutated_individual

# 选择最优个体
def select_best(population, fitnesses, num_best=seed//2):
    indices = np.argsort(fitnesses)[-num_best:]
    return population[indices]

# 种群更新函数
def evolution_strategy_update(population, fitnesses, current_step):
    best_individuals = select_best(population, fitnesses)
    new_population = []
    
    for _ in range(len(population)):
       
        parent = best_individuals[np.random.randint(len(best_individuals))]
        child = mutation(parent, current_step)
        new_population.append(child)
    
    return np.array(new_population)

# 设定置信度阈值
CONFIDENCE_THRESHOLD = 0.5  

def run_attack():
    
    img_path = '/root/autodl-tmp/code/val_people/val_people/FLIR_08875.jpeg'  
    
    
    initial_bboxes, initial_confidences = detection(cv2.imread(img_path), model_car_inf_yolov3)
   
    if len(initial_bboxes) != 1:
        print(f"跳过图像: {img_path}, 检测到 {len(initial_bboxes)} 个行人")
        return

   
    print(f"图像: {img_path}，初始置信度: {initial_confidences[0]}")

    X1, Y1, X2, Y2 = map(int, initial_bboxes[0][:4])  
    population = initialize_population(X1, Y1, X2, Y2)  
    P_best = population.copy()
    G_best = population[0]
    conf_G = 100
    conf_P = [100] * seed

    for steps in range(step):
        success = False  
        fitnesses = []

        for i in range(seed):
            perturbed_image_path = 'temp_adv.jpg'

           
            perturbed_image = cv2.imread(img_path)
            add_catmull_rom_curve(perturbed_image, [initial_bboxes[0]])  
            cv2.imwrite(perturbed_image_path, perturbed_image)  

            
            res_inf = yolov3_inf(perturbed_image_path, model_car_inf_yolov3)
            
            if res_inf:
                attacked_confidence = res_inf[0][4]
                fitnesses.append(fitness_function(attacked_confidence))
                print(f"图像: {img_path}，攻击后置信度: {attacked_confidence}")
            else:
                fitnesses.append(fitness_function(0))  
                print(f"图像: {img_path}，攻击后未检测到行人")
            
           
            if not res_inf or (res_inf[0][4] < CONFIDENCE_THRESHOLD):
                success = True
                print(f"攻击成功：{img_path}")
                break

            # 更新最优位置
            conf = res_inf[0][4] if res_inf else 0  
            if conf < conf_G:  
                conf_G = conf
                G_best = population[i].copy()

            if conf < conf_P[i]:
                P_best[i] = population[i].copy()
                conf_P[i] = conf 

        if success:
            break

      
        population = evolution_strategy_update(population, np.array(fitnesses), steps)
        
        
        for i in range(seed):
            population[i] = clip_catmull_rom_points(population[i], X1, Y1, X2, Y2)




run_attack()
