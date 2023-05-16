import pickle as pkl
import numpy as np
import operator
import copy
import os
# 块的数量
block_num = 20
# 每个块中节点数目
block_node_num = 0
# 最后一个块中节点数目
final_block_node_num = 0
# 节点总数
node_num = 0
convergence_threshold = 0.00001

def read_data(path):
    """
    从 txt 中读取数据 分块 保存到本地
    :param path: data.txt path
    :return:
    """
    # 节点总数
    global node_num
    global block_node_num, final_block_node_num

    n2n_dic = {}
    #n2n_dic[0] = []
    # with open("data/Data.txt") as file:
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            split = line.split()
            l_node_idx = int(split[0])
            r_node_idx = int(split[1])
            # node 总数
            node_num = max(node_num, l_node_idx, r_node_idx)
            if l_node_idx not in n2n_dic.keys():
                # 出度数
                n2n_dic[l_node_idx] = [1, []]
                # 指向节点列表
                n2n_dic[l_node_idx][1] = [r_node_idx]
                #print("add an item: n2n_dic[", l_node_idx, "]=" ,n2n_dic[l_node_idx])
            else:
                n2n_dic[l_node_idx][0] += 1
                n2n_dic[l_node_idx][1].append(r_node_idx)
    f.close()
    # block 中节点数目
    block_node_num = node_num // block_num + 1
    final_block_node_num = node_num % block_node_num
    # test
    if (block_num-1)*block_node_num + final_block_node_num == node_num:
        print("cut block ok")
    # print(n2n_dic)
    # 按 key 排序
    n2n_dic = dict(sorted(n2n_dic.items(), key=operator.itemgetter(0)))
    #n2n_dic = sorted(n2n_dic.keys())
    #print("afer sort", n2n_dic)
    print(f"block_num：{block_num},node_num:{node_num}, block_node_num:{block_node_num}, final_block_node_num:{final_block_node_num}")
    #return
    # save matrix to local
    dirs = "block_pg1/mid_res"
    if not os.path.exists(dirs):
        os.makedirs(dirs)
    matrix_path = "block_pg1/mid_res/block_"
    #block_n2n_dic = {}
    for i in range(0, block_num):
        #print(n2n_dic)
        block_n2n_dic = copy.deepcopy(n2n_dic)
        # 复制后的 dic 修改会影响原来的 dic
        #block_n2n_dic = n2n_dic.copy()
        # for k in n2n_dic:
        #     """
        #     实际上，分块，不是按照dic分的，而是按照 score 分的，因为所有节点都要有得分，所以我们把score分成好几个列表
        #     更新 score 时需要涉及到这个score列表节点中的所有节点
        #     所以，这个块，就是将指向score节点的节点汇总了起来
        #     """
        #     # 保证这个块中的节点，它的出边只有 block_node_num 个
        #     # block_n2n_dic 中只有有出边的节点，这个筛选条件保证这个块中存储的哪些节点，节点的指向节点只含有第几个块score中的那些节点
        #     # 删掉了没用的，因为计算的时候是遍历dic的指向节点，而我们计算这个score块的，不能score超出范围，所以在这里删了，但出度数不能变
        #     # block_n2n_dic[k] = [n2n_dic[k][0],
        #     #                                 [ele for ele in n2n_dic[k][1] if ele < (i+1) * block_node_num and ele >= i * block_node_num]]
        #     block_n2n_dic[k] = n2n_dic[k]
        #print(f"start block_{i}:{block_n2n_dic}")
        # 删掉用不到的
        #tmp_keys =
        for key in list(block_n2n_dic.keys()):
            #print(key, " ", block_n2n_dic[key])
            #tmp_list = []
            block_n2n_dic[key][1] = list(
                filter(lambda idx: idx >= i * block_node_num and idx < (i + 1) * block_node_num, block_n2n_dic[key][1]))
            # for idx in block_n2n_dic[key][1]:
            #     if idx < i * block_node_num or idx >= (i+1)*block_node_num:
            #         print(f"idx[{idx}] < i * block_node_num[{i * block_node_num}] or  >= (i+1)*block_node_num[{(i+1)*block_node_num}]")
            #         print(f"remove {idx} from {block_n2n_dic[key]}")
            #         block_n2n_dic[key][1].remove(idx)
            if block_n2n_dic[key][1] == []:
                #print(f"key = {key} is blank block_{i}:del {block_n2n_dic[key]}")
                del block_n2n_dic[key]
        #print(f"final block_{i}:{block_n2n_dic}")
        # save to loacl
        f = open(matrix_path + str(i), "wb")
        pkl.dump(block_n2n_dic, f)
        print(f"save block_{i}!")
        f.close()
    print("now read data finish.")


def block_cal_pg(K, ct):
    """
    block PageRank 计算
    :param K: 阻尼因子
    :param ct: 最大循环次数
    :return:
    """
    print("-----cal pagerank.")

    global node_num
    global block_node_num, final_block_node_num
    # 初始化得分列表
    # 不用idx为0的元素，node num+1
    score_ini = [1 / float(node_num)] * (node_num + 1)
    score_ini[0] = 0
    #print(score_ini)
    round = 0
    while True:
        last_score = score_ini.copy()
        # 分块计算
        for i in range(0, block_num):
            block_file = open("block_pg1/mid_res/block_" + str(i), "rb")
            block_n2n_dic = pkl.load(block_file)
            block_file.close()
            #print(f"load data block_{i}:{block_n2n_dic}")
            for key, value in block_n2n_dic.items():
                #if key >= i * block_node_num and key < (i + 1) * block_node_num:
                for to_node_idx in value[1]:
                    score_ini[to_node_idx] += last_score[key]/float(value[0])
        # normalize
        sum_score = sum(score_ini)
        # print(sum(score_ini))
        # print("normalize")
        for j in range(1, node_num + 1):
            score_ini[j] = score_ini[j] / sum_score

        #print("update score", score_ini)
        # add K
        # print("add k")
        dif_sum = 0.0
        for j in range(0, node_num + 1):
            if j == 0:
                continue
            score_ini[j] = score_ini[j] * K + (1 - K) / float(node_num)
            dif_sum += abs(score_ini[j] - last_score[j])
            # print(dif_sum)
        print("dif_sum =", dif_sum)
        # print(score_ini)
        # 达到最大轮数
        if dif_sum <= convergence_threshold or round == ct:
            print("stop")
            #print(sum(score_ini))
            # print("收敛 dif_sum = ", dif_sum)
            # print(score_ini)
            break
        else:
            last_score = score_ini
        round += 1

    result = np.array(score_ini)
    sort_result = dict(zip(np.argsort(-result)[:100], sorted(result, reverse=True)[:101]))
    # print(sort_result)
    l1 = []
    with open("block_result.txt", "w") as f:
        for key in sort_result:
            l1.append(key)
            f.write(str(key) + '\t' + str(sort_result[key]) + '\n')
    return  l1

# def test(l1):
#     """
#     调用库函数测试结果差异
#     :return:
#     """
#     import networkx as nx
#     # 读入有向图，存储边
#     f = open("Data.txt", 'r')
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
#     #print(h1)
#     count = 0
#     list_100 = []
#     for id, score in h1.items():
#        count += 1
#        if count == 100:
#            break
#        list_100.append(id)
#
#     num = 0
#     for i in range(len(l1)):
#         if l1[i] in list_100:
#             continue
#         else:
#             num += 1
#     print(num)
#
#        # str_save = str(id) + " " + str(score)
#        # with open(r'test_res1.txt', 'a') as f:
#        #     f.write(str_save)
#        #     f.write('\n')
#     #f.close()

if __name__ == '__main__':
    data_path = "Data.txt"
    #data_path = "data.txt"
    read_data(data_path)
    # print(f"block_num：{block_num},node_num:{node_num}, block_node_num:{block_node_num}, final_block_node_num:{final_block_node_num}")
    max_round = 50
    alphe = 0.75
    l1 = block_cal_pg(alphe, max_round)

    # 检测 block 算法和 nx 差异
    # test(l1)

    # dic = {1: [1, [2]], 2: [3, [1]], 3: [1, [4]], 4: [1, [2]], 7: [1, [1]], 8: [2, [2]], 9: [2, [3]]}
    # d2 = dict(filter(
    #     lambda x: x[0] >= 0 and x[0] < 5,
    #     dic.items()))
    # print(d2)
    # d1 = {1:"a"}
    # d2 = d1
    # d3 = d1.copy()
    # print(f"d2{d2}, d3{d3}")
    # d1[2] = '0'
    # print(f"d2{d2}, d3{d3}")
    # del d3[1]
    # print(f"d1 {d1} d2{d2} d3{d3}")
    # l = [1,2,3]
    # l1 = l.copy()
    # print(l1)
    # l[2] = 99
    # print(l1)



