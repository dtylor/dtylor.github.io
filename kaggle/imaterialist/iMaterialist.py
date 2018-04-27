from fastai.conv_learner import *
from fastai.plots import *
from planet import f2

if __name__ == '__main__':
 PATH = 'data/imaterialist/'
 metrics=[f2]
 f_model = resnet34
 label_csv = f'{PATH}train.csv'
 n = len(list(open(label_csv)))-1
 val_idxs = get_cv_idxs(n)	
 def get_data(sz):
  tfms = tfms_from_model(f_model, sz, aug_tfms=transforms_side_on, max_zoom=1.05)
  return ImageClassifierData.from_csv(PATH,'train',label_csv,tfms=tfms,val_idxs=val_idxs,test_name='test')
 sz=64
 data = get_data(sz)
 learn = ConvLearner.pretrained(f_model, data, metrics=metrics)
 #lrf=learn.lr_find()
 #learn.sched.plot()
 lr = .015
 learn.fit(lr,2, cycle_len=1, cycle_mult=2)
 learn.save(f'{sz}')
 lrs = np.array([lr/9,lr/3,lr])
 learn.unfreeze()
 learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)
 learn.save(f'{sz}')
 
 sz=128
 learn.set_data(get_data(sz))
 learn.freeze()
 learn.fit(lr, 2, cycle_len=1, cycle_mult=2)
 learn.save(f'{sz}')
 learn.unfreeze()
 learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)
 learn.save(f'{sz}')

 sz=300
 learn.set_data(get_data(sz))
 learn.freeze()
 learn.fit(lr, 2, cycle_len=1, cycle_mult=2)
 learn.save(f'{sz}')
 learn.unfreeze()
 learn.fit(lrs, 2, cycle_len=1, cycle_mult=2)
 multi_preds, y = learn.TTA()
 preds = np.mean(multi_preds, 0)
 learn.save(f'{sz}')
 f2(preds,y)

