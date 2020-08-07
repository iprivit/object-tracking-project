if __name__ == "__main__":
    dataloader = {'vis_threshold':0.3, 'P':18, 'K':4, 'max_per_person':1000, 'crop_H':256, 'crop_W':128, 
                 'transform':'random', 'normalize_mean':[0.485, 0.456, 0.406], 'normalize_std':[0.229, 0.224, 0.225] }

    # seqclass = SeqDataClass()
    # helmreid = helmreID(vid='57584_002767_Sideline', split='train', vis_threshold=0.0, P=18, K=4, max_per_person=1000, 
    #         crop_H=256, crop_W=128, transform='random', normalize_mean=[0.485, 0.456, 0.406],
    #                  normalize_std=[0.229, 0.224, 0.225])
    print('progress check')
    if not multihelm:
        multihelm = helmreIDWrapper('train',dataloader)
    print('progress check')
    db_train = DataLoader(multihelm, batch_size=1, shuffle=True, num_workers=4)

    output_dir = osp.join(get_output_dir('reid'), 'test_finetune_orig') # reid['name']
    tb_dir = osp.join(get_tb_dir('reid'), 'test_finetune_orig') # reid['name']

    # db_train = Datasets(reid['db_train'], reid['dataloader'])
    # db_train = DataLoader(db_train, batch_size=1, shuffle=True)

    # if reid['db_val']:
    #     db_val = None
    #     #db_val = DataLoader(db_val, batch_size=1, shuffle=True)
    # else:
    #     db_val = None

    db_val = None

    ##########################
    # Initialize the modules #
    ##########################
    print("[*] Building CNN")
    network = resnet50(pretrained=True, output_dim=128) #**reid['cnn'])   cnn:output_dim: 128
    network.load_state_dict(torch.load( '/opt/ml/tracking_wo_bnw/output/tracktor/reid/test/ResNet_iter_25137.pth', #tracktor['reid_weights'],
                                 map_location=device))
    # /home/ubuntu/code/tracking_wo_bnw/output/tracktor/reid/test2/ResNet_iter_8360.pth
    # /home/ubuntu/code/tracking_wo_bnw/output/tracktor/reid/test_finetune_orig/ResNet_iter_440.pth
    network.train()
    network.cuda()

    ##################
    # Begin training #
    ##################
    print("[*] Solving ...")

    # build scheduling like in "In Defense of the Triplet Loss for Person Re-Identification"
    # from Hermans et al.
    lr = 0.0003 #reid['solver']['optim_args']['lr']
    iters_per_epoch = len(db_train)
    # we want to keep lr until iter 15000 and from there to iter 25000 a exponential decay
    l = eval("lambda epoch: 1 if epoch*{} < 15000 else 0.001**((epoch*{} - 15000)/(25000-15000))".format(
                                                                iters_per_epoch,  iters_per_epoch))

    model_args = {'loss':'batch_hard','margin':0.2, 'prec_at_k':3}

    max_epochs = 25000 // len(db_train.dataset) + 1 if 25000 % len(db_train.dataset) else 25000 // len(db_train.dataset)
    solver = Solver(output_dir, tb_dir, lr_scheduler_lambda=l)
    solver.train(network, db_train, db_val, max_epochs, 100, model_args=model_args) #reid['model_args'])