import kagglehub

# Download latest version
path = kagglehub.dataset_download("alifrahman/chestxraydataset")

print("Path to dataset files:", path)