k2
    # with open('./test_offline/task2_gt.json','r') as f:
    #     task2_gt = json.load(f)
    # task2_pred = test_task2('./test_offline/task2')
    # task2_acc = utils.calc_accuracy(task2_gt,task2_pred)
    # print('accuracy for task2 is:',task2_acc)   

    # # ## testing task3
    # test_task3('./test_offline/task3','./test_offline/task3_estimate')
    # task3_SISDR_blind = utils.calc_SISDR('./test_offline/task3_gt','./test_offline/task3_estimate',permutaion=True)  # 盲分离
    # print('strength-averaged SISDR_blind for task3 is:',task3_SISDR_blind)
    # task3_SISDR_match = utils.calc_SISDR('./test_offline/task3_gt','./test_offline/task3_estimate',permutaion=False) # 定位分离
    # print('strength-averag