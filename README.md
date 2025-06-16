# faster_RCNN
I am not uploading my full code here for now, just the overview.
Thanks

________________________________________
1. Image Transformations
def get_transform(train):
    return T.Compose([
        T.Resize((height_, width_)),
        T.Lambda(lambda x: x.repeat(3, 1, 1)),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
•	Resizes image to fixed dimensions (height = 320, width = 320)
•	Converts grayscale to 3 channels
•	Applies normalization
________________________________________
2. Padding Targets for Batching
def pad_target(target, max_num_boxes, padding_label=0):
    ...
    padded_boxes = torch.cat([target['boxes'], padding_boxes], dim=0)
    padded_labels = torch.cat([target['labels'], padding_labels], dim=0)
•	Ensures all targets have same number of boxes
•	Pads with dummy boxes and labels if needed
________________________________________
3. Custom Collate Function
def custom_collate_fn(batch):
    ...
    max_num_boxes = max([item[1]['boxes'].size(0) for item in valid_batch_items])
    targets = [pad_target(item[1], max_num_boxes) for item in batch]
•	Filters out invalid entries
•	Applies padding logic across a batch
________________________________________
4. Custom Dataset with Augmentations
class CustomDataset(Dataset):
    def __getitem__(self, idx):
        ...
        scale_factor = random.uniform(*self.scale_range)
        image = image.resize((new_width, new_height))
        boxes *= scale_factor
        ...
        if random.random() < self.rotation_prob:
            image, boxes = self.rotate_image_and_boxes(image, boxes, angle)
        ...
        return noisy_image, target
•	Loads image and boxes
•	Applies random scale and rotation
•	Normalizes, adds Gaussian noise
•	Filters invalid boxes
________________________________________
5. Anchor Box Calculation with KMeans
kmeans = KMeans(n_clusters=5, random_state=0, n_init=10).fit(wh_array)
anchor_boxes = kmeans.cluster_centers_
anchor_sizes = tuple([(int(round(np.sqrt(w * h))),) for w, h in anchor_boxes])
aspect_ratios = tuple([tuple((w / h,) for w, h in anchor_boxes)] * len(anchor_sizes))
•	Clusters box sizes to get representative anchor sizes and ratios
•	Converts to format compatible with AnchorGenerator
________________________________________
6. Model Setup with Attention and Custom Predictor
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
model.backbone = add_cbam_spatial_attention(model.backbone)
model.roi_heads.box_predictor = WeightedFastRCNNPredictor(in_features, num_classes, class_weights)
•	Loads Faster R-CNN with MobileNet v3 backbone
•	Adds spatial attention to feature layers
•	Replaces classification head with a custom one using class weights
________________________________________
7. Training Loop with Gradient Scaling
for epoch in range(last_epoch, num_epochs):
    for batch_idx, (images, targets) in enumerate(train_loader):
        ...
        with autocast(device_type='cuda'):
            loss_dict = model(images, targets)
            total_loss = sum(loss for loss in loss_dict.values()) + l1_lambda * l1_reg
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()
•	Uses mixed-precision training for efficiency
•	Computes total loss with L1 regularization
•	Applies scaling with GradScaler
________________________________________
8. Saving Checkpoints and Learning Rate Scheduling
if (epoch + 1) % 5 == 0:
    torch.save(model, base_path + f'epoch_{epoch + 1}.pth')

if (epoch + 1) % 7 == 0:
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_reduction_factor
•	Saves model every few epochs
•	Reduces learning rate periodically
________________________________________

