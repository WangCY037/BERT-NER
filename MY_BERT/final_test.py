
import torch

from MY_BERT.model import my_model
from MY_BERT.main import eval,prepare_dataloaders

def prepare(checkpoint):
    checkpoint=torch.load(checkpoint)
    model_state_dict=checkpoint['model']

    opt=checkpoint['settings']
    #========= Loading Dataset =========#
    data = torch.load(opt.data)

    train_loader,valid_loader,test_loader = prepare_dataloaders(data, opt)



    #========= Preparing Model =========#

    print(opt)

    device = torch.device('cuda' if opt.cuda else 'cpu')
    bert_classfication_model = my_model(opt.out_classes).to(device)
    bert_classfication_model.load_state_dict(model_state_dict)

    return bert_classfication_model,train_loader,valid_loader,test_loader,device

def eval_final(transformer,train_loader,valid_loader,test_loader,device):
    train_loss, train_f1, train_acc=eval(transformer,train_loader,device)
    valid_loss, valid_f1, valid_acc = eval(transformer, valid_loader, device)
    test_loss, test_f1, test_acc = eval(transformer, test_loader, device)
    print('train:loss {loss:.3f} f1 {f1:.3f} acc {acc:.3f}'.format(loss=train_loss,f1=100*train_f1,acc=100*train_acc))
    print('valid:loss {loss:.3f} f1 {f1:.3f} acc {acc:.3f}'.format(loss=valid_loss,f1=100*valid_f1,acc=100*valid_acc))
    print('test:loss {loss:.3f} f1 {f1:.3f} acc {acc:.3f}'.format(loss=test_loss,f1=100*test_f1,acc=100*test_acc))

def main():
    checkpoint='save_model.chkpt'
    transformer, train_loader, valid_loader, test_loader, device=prepare(checkpoint)
    eval_final(transformer,train_loader,valid_loader,test_loader,device)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()

    # - (Training)   loss:  0.00001, accu: 98.555 %, f1: 90.580 %,elapse: 0.575 min
    # - (Validation)   loss:  0.00001, accu: 98.741 %, f1: 94.971 %,elapse: 0.058 min
    #   - [Info] The checkpoint file has been updated.

