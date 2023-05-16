import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
import operator

convergence_threshold = 0.00001

def read_data(data_path):
    """
    :param data_path: data路径
    :return: 所有节点数目，字典dic[from] = [all to nodes]
    """
    node_num = 0
    n2n_dic = {}
    #n2n_dic[0] = list()
    with open(data_path) as f:
        lines = f.readlines()
        for line in lines:
            id_2_str = line.split()
            #print(type(split[0]))
            l_node_idx = int(id_2_str[0])
            r_node_idx = int(id_2_str[1])
            #print("from node", l_node_idx, "to node", r_node_idx)
            # 节点总数
            node_num = max(node_num, l_node_idx, r_node_idx)
            #print("now the max node idx is ", node_num)
            if l_node_idx not in n2n_dic.keys():
                n2n_dic[l_node_idx] = [r_node_idx]
                #print("add an item: n2n_dic[", l_node_idx, "]=" ,n2n_dic[l_node_idx])
            else:
                n2n_dic[l_node_idx].append(r_node_idx)
            #print(n2n_dic)
    f.close()

    print("all nodes num is ", node_num)
    #print(n2n_dic)
    # 保存
    # f_save = open("Matrix.matrix", "wb")
    # pkl.dump(n2n_dic, f_save)
    # f_save.close()
    # print("save n2n_dic to local.")
    return node_num, n2n_dic


def my_pagerank(node_num, n2n_dic, K, count):
    """
    basic pagerank 计算
    :param node_num: 节点总数
    :param n2n_dic: 所有出链节点及其链接关系
    :param K: 阻尼因子 0.85
    :param count: 最大循环次数
    :return:
    """
    # 初始化得分列表
    # 不用idx为0的元素，node num+1
    score_ini = [1/float(node_num)] * (node_num + 1)
    score_ini[0] = 0
    #print("ini score list:", score_ini)
    last_score = score_ini.copy()
    # 最多计算 count 轮
    i = 0
    while True:
    #for i in range(0, count):
        print("round", i)
        # if i == 1:
        #     print(node_num)
        #last_score = score_ini.copy()
        # cal
        # print("cal")
        for from_node, to_list in n2n_dic.items():
            #print(to_list)
            to_num = len(to_list)
            for to_node in to_list:
                score_ini[to_node] += last_score[from_node]/float(to_num)
        # 归一化
        sum_score = sum(score_ini)
        # print("normalize")
        for j in range(1, node_num+1):
            score_ini[j] = score_ini[j]/sum_score
        # print(score_ini)
        # print(sum(score_ini))
        # 阻尼因子和随机跳转概率
        # print("add k")
        dif_sum = 0.0 # 前后两轮差值
        for j in range(0, node_num+1):
            if j == 0:
                continue
            score_ini[j] = score_ini[j] * K + (1 - K)/float(node_num)
            dif_sum += abs(score_ini[j] - last_score[j])
        #print("dif_sum =", dif_sum)
        #print(score_ini)
        # 轮数+1
        i = i + 1
        if dif_sum <= convergence_threshold or i == count:
            #print(f"stop round = {i}")
            #print(sum(score_ini))
            # print("收敛 dif_sum = ", dif_sum)
            # print(score_ini)
            print(f"sum(score_ini) is {sum(score_ini)} and test score[0] is {score_ini[0]}")
            break
        else:
            last_score = score_ini

        #break
    # 排序
    result = np.array(score_ini)
    sort_result = dict(zip(np.argsort(-result)[:100], sorted(result, reverse=True)[:101]))
    #print(sort_result)
    l1 = []
    print("------------------output res-------------------------")
    with open("basic_result.txt", "w") as f:
        for key in sort_result:
            f.write(str(key) + ' ' + str(sort_result[key]) + '\n')
            #l1.append(key)
            print(str(key) + ' ' + str(sort_result[key]))
    #return l1

# def test():
#     """
#     调用库函数测试结果差异
#     :return:
#     """
#     # 读入有向图，存储边
#     f = open("data/Data.txt", 'r')
#     lines = f.readlines()
#     edges = []
#     for line in lines:
#         split = line.split()
#         # print(type(split[0]))
#         l_node_idx = int(split[0])
#         r_node_idx = int(split[1])
#         edges.append([l_node_idx, r_node_idx])
#
#     # edges = [line.strip('\n').split(' ') for line in f]
#     # print(edges)
#     G = nx.DiGraph()
#     for edge in edges:
#         G.add_edge(edge[0], edge[1])
#     pg = nx.pagerank(G, alpha = 0.85,tol=0.00001)
#
#     h = sorted(pg.items(), key=lambda e: e[1])
#     h1 = {}
#     for i in h:
#         h1[i[0]] = i[1]
#     # print(h1)
#     h1 = dict(sorted(h1.items(), key=operator.itemgetter(1), reverse = True))
#     # h2 = sorted(h1.items(), key=lambda x: x[1])
#     print(h1)
#     count = 0
#     list_100 = []
#     for id, score in h1.items():
#        count += 1
#        if count == 100:
#            break
#        list_100.append(id)
#
#     return list_100
#
#        # str_save = str(id) + " " + str(score)
#        # with open(r'test_res1.txt', 'a') as f:
#        #     f.write(str_save)
#        #     f.write('\n')
#     #f.close()

if __name__ == '__main__':
    print("--------------------------read data--------------------")
    path = "data/Data.txt"
    #path = "data.txt"
    all_node_num, node2node_dic = read_data(path)
    print("--------------------------cal pagerank--------------------")
    K = 0.85
    ct = 50
    my_pagerank(all_node_num, node2node_dic, K, ct)
    #l1 = my_pagerank(all_node_num, node2node_dic, K, ct)
    #print("============================================")


    # l2 = test()
    # num = 0
    # for i in range(len(l1)):
    #     if l1[i] in l2:
    #         continue
    #     else:
    #         num += 1
    # print(num)