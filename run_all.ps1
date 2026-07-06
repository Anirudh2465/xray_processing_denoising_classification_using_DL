python -u src\train_classifier.py --model unet --epochs 3
python -u src\train_classifier.py --model resnet --epochs 3
python -u src\train_classifier.py --model densenet --epochs 3
python -u src\evaluate_classifiers.py
