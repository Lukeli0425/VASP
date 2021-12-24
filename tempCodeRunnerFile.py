r') as f:
        task2_gt = json.load(f)
    task2_pred = test_task2('./test_offline/task2')
    task2_acc = utils.calc_accuracy(task2_gt,task2_pred)
    print('accuracy for task2 is:',task2_acc)  