import os
import shutil


def prepare_tiny_imagenet_val(base_path='dataset/tiny-imagenet-200'):
    val_annotations = os.path.join(base_path, 'val/val_annotations.txt')
    images_dir = os.path.join(base_path, 'val/images')
    
    # Check if already reorganized
    if not os.path.exists(images_dir):
        print(f"Validation set already reorganized")
        return
    
    print(f"Reorganizing validation set in {base_path}...")
    
    with open(val_annotations) as f:
        for line in f:
            fn, cls, *_ = line.split('\t')
            
            # Create class directory
            class_dir = os.path.join(base_path, f'val/{cls}')
            os.makedirs(class_dir, exist_ok=True)
            
            # Copy image to class directory
            src = os.path.join(images_dir, fn)
            dst = os.path.join(class_dir, fn)
            
            if os.path.exists(src):
                shutil.copyfile(src, dst)
    
    # Remove old images directory
    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
        print(f"Validation set reorganized successfully")


if __name__ == "__main__":
    prepare_tiny_imagenet_val()