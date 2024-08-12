import numpy as np
import cv2



def id_num_matching(ocr_results, cluster_results, match_dict):
    index=0
    for ocr_result in ocr_results:
        try:
            nums = np.array(ocr_result.cls)
            conf= np.array(ocr_result.conf)
            id=str(cluster_results[index][5])
            if nums.shape==(1,): ## 한개의 클래스만 인식된 경우
                num_1=int(nums.item())
                conf_1=np.round(float(conf.item()),4)
                if id not in match_dict:
                    cluster_results[index].append(num_1)
                    match_dict[id] = {}
                    match_dict[id]['num']=num_1
                    match_dict[id]['conf']=conf_1
                elif id in match_dict:
                    if match_dict[id]['conf']>=conf_1:
                        cluster_results[index].append(match_dict[id]['num'])
                    else:
                        cluster_results[index].append(num_1)
                        match_dict[id]['num']=num_1
                        match_dict[id]['conf']=conf_1
                index+=1
            elif nums.shape==(2,): ## 두개의 클래스가 인식된 경우
                position=ocr_result.boxes.xywh.cpu().numpy()
                position_2D = np.reshape(position, (-1, 4))
                position_x=[]
                position_x.append(position_2D[0,0])
                position_x.append(position_2D[1,0])
                if abs(position_x[0]-position_x[1])>2:   #동일한 위치에 두번 인식되지 않은 경우
                    conf_2=np.round(float(np.max(conf)),4)
                    if position_x[0]<position_x[1]:
                        num_2 = int(''.join(map(lambda x: str(int(x)), nums)))
                    elif position_x[0]>position_x[1]:
                        num_2 = int(''.join(map(lambda x: str(int(x)), nums[::-1])))
                    else:
                        continue 
                    if id not in match_dict:
                        cluster_results[index].append(num_2)
                        match_dict[id] = {}
                        match_dict[id]['num']=num_2
                        match_dict[id]['conf']=conf_2
                    elif id in match_dict:
                        if match_dict[id]['conf']>=conf_2:
                            cluster_results[index].append(match_dict[id]['num'])
                        else:
                            cluster_results[index].append(num_2)
                            match_dict[id]['num']=num_2
                            match_dict[id]['conf']=conf_2
                else:      #동일한 위치에 두 번 인식된 경우
                    conf_arr=np.reshape(conf,(-1,1))
                    num_arr=np.reshape(nums,(-1,1))
                    if conf_arr[0,0]>conf_arr[1,0]: #confidence 값이 더 큰 숫자를 사용
                        num_2=int(num_arr[0,0])
                        conf_2=np.round(float(np.max(conf)),4)
                    else:
                        num_2=int(num_arr[1,0])
                        conf_2=np.round(float(np.max(conf)),4)
                    if id not in match_dict:
                        cluster_results[index].append(num_2)
                        match_dict[id]['num']=num_2
                        match_dict[id]['conf']=conf_2
                    elif id in match_dict:
                        if match_dict[id]['conf']>=conf_2:
                            cluster_results[index].append(match_dict[id]['num'])
                        else:
                            cluster_results[index].append(num_2)
                            match_dict[id]['num']=num_2
                            match_dict[id]['conf']=conf_2
                index+=1
            else:
                if id not in match_dict:
                    cluster_results[index].append(int(100))
                elif id in match_dict:
                    cluster_results[index].append(match_dict[id]['num'])
                index+=1
        except:
            if id not in match_dict:
                cluster_results[index].append(int(100))
            elif id in match_dict:
                cluster_results[index].append(match_dict[id]['num'])
            index+=1
    return match_dict, cluster_results

