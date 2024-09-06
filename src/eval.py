import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import matplotlib.animation as animation


from new_pp import Segmenter, MedicalImageDataset, val_loader

model = Segmenter.load_from_checkpoint('/home/ubuntu/liver/src/logs/lightning_logs/version_1/checkpoints/epoch=9-step=490.ckpt')
# model.eval()

#
# def animate_slices(image, mask, prediction, interval=200):
#     """
#     Creates an animation of slices through a 3D image, ground truth mask, and predicted mask.
#
#     Args:
#     - image (torch.Tensor): The input image tensor.
#     - mask (torch.Tensor): The ground truth mask tensor.
#     - prediction (torch.Tensor): The predicted mask tensor.
#     - interval (int): Interval between frames in milliseconds.
#     """
#     fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#
#     slice_idx = image.shape[1]  # Number of slices in the 3D volume
#
#     def update(i):
#         ax[0].clear()
#         ax[1].clear()
#         ax[2].clear()
#
#         ax[0].imshow(image[0, i, :, :], cmap='gray')
#         ax[0].set_title(f'Image - Slice {i + 1}')
#
#         ax[1].imshow(mask[i, :, :], cmap='gray')
#         ax[1].set_title(f'Ground Truth Mask - Slice {i + 1}')
#
#         ax[2].imshow(prediction[i, :, :], cmap='gray')
#         ax[2].set_title(f'Predicted Mask - Slice {i + 1}')
#
#     ani = animation.FuncAnimation(fig, update, frames=slice_idx, interval=interval)
#     plt.show()
#
#
# def evaluate_and_animate(model, val_loader, device='cuda'):
#     """
#     Evaluates the model on the validation set and visualizes the predictions with animation.
#
#     Args:
#     - model: The trained segmentation model.
#     - val_loader: DataLoader for validation set.
#     - device: Device to run the evaluation on ('cuda' or 'cpu').
#     """
#     model.eval()  # Set model to evaluation mode
#     model.to(device)
#
#     with torch.no_grad():
#         for batch_idx, (images, masks) in enumerate(val_loader):
#             images, masks = images.to(device), masks.to(device)
#
#             # Forward pass
#             outputs = model(images)
#
#             # Apply softmax and take argmax to get the predicted mask
#             preds = torch.argmax(outputs, dim=1).cpu()
#
#             # Get the ground truth mask (detach and move to CPU for visualization)
#             masks = masks.cpu()
#
#             # Visualize for the first image in the batch
#             image = images[0].cpu().numpy()
#             mask = masks[0].numpy()
#             pred = preds[0].numpy()
#
#             # Animate the entire 3D volume
#             animate_slices(image, mask, pred)
#
#             # Exit after visualizing one batch to prevent excessive output
#             break


# Function to calculate Dice coefficient
def calculate_dice_coefficient(preds, target, epsilon=1e-6):
    """
    Calculates the Dice coefficient (F1 score) for each slice in the 3D volume.

    Args:
    - preds (torch.Tensor): The predicted mask tensor.
    - target (torch.Tensor): The ground truth mask tensor.
    - epsilon (float): Small constant to avoid division by zero.

    Returns:
    - dice_scores (list): List of Dice coefficients for each slice.
    """
    dice_scores = []
    for i in range(target.shape[0]):  # Iterate through each slice in the 3D volume
        intersection = (preds[i] * target[i]).sum().item()
        union = preds[i].sum().item() + target[i].sum().item()
        dice = (2. * intersection + epsilon) / (union + epsilon)
        dice_scores.append(dice * 100)  # Multiply by 100 for percentage
    return dice_scores


# Function to visualize 2D slices of 3D images with animation and display Dice score
def animate_slices_with_dice(image, mask, prediction, dice_scores, interval=200):
    """
    Creates an animation of slices through a 3D image, ground truth mask, and predicted mask,
    and displays the Dice coefficient.

    Args:
    - image (torch.Tensor): The input image tensor.
    - mask (torch.Tensor): The ground truth mask tensor.
    - prediction (torch.Tensor): The predicted mask tensor.
    - dice_scores (list): Dice coefficients for each slice.
    - interval (int): Interval between frames in milliseconds.
    """
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    slice_idx = image.shape[1]  # Number of slices in the 3D volume

    def update(i):
        ax[0].clear()
        ax[1].clear()
        ax[2].clear()

        ax[0].imshow(image[0, i, :, :], cmap='gray')
        ax[0].set_title(f'Image - Slice {i + 1}')

        ax[1].imshow(mask[i, :, :], cmap='gray')
        ax[1].set_title(f'Ground Truth Mask - Slice {i + 1}')

        ax[2].imshow(prediction[i, :, :], cmap='gray')
        ax[2].set_title(f'Predicted Mask - Slice {i + 1} \nDice Score: {dice_scores[i]:.2f}%')

    ani = animation.FuncAnimation(fig, update, frames=slice_idx, interval=interval)
    plt.show()


# Updated evaluation loop with Dice coefficient
def evaluate_and_animate_with_dice(model, val_loader, device='cuda'):
    """
    Evaluates the model on the validation set, calculates Dice coefficient, and visualizes the predictions with animation.

    Args:
    - model: The trained segmentation model.
    - val_loader: DataLoader for validation set.
    - device: Device to run the evaluation on ('cuda' or 'cpu').
    """
    model.eval()  # Set model to evaluation mode
    model.to(device)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)

            # Apply softmax and take argmax to get the predicted mask
            preds = torch.argmax(outputs, dim=1).cpu()

            # Get the ground truth mask (detach and move to CPU for visualization)
            masks = masks.cpu()

            # Visualize for the first image in the batch
            image = images[0].cpu().numpy()
            mask = masks[0].numpy()
            pred = preds[0].numpy()

            # Calculate Dice coefficient per slice
            dice_scores = calculate_dice_coefficient(pred, mask)

            # Animate the entire 3D volume with Dice scores
            animate_slices_with_dice(image, mask, pred, dice_scores)

            # Exit after visualizing one batch to prevent excessive output
            break


# Assuming the model is already trained
evaluate_and_animate_with_dice(model, val_loader)







#
# def visualize_predictions(model, val_loader, num_samples=4):
#     model.eval()
#     fig, axs = plt.subplots(num_samples, 3, figsize=(10, num_samples * 3))
#
#     with torch.no_grad():
#         for i, (img, mask) in enumerate(val_loader):
#             if i == num_samples:
#                 break
#             img = img.cuda()  # Assuming you're using GPU
#             mask = mask.cuda()
#
#             # Get the model prediction
#             y_hat = model(img)
#             y_hat_softmax = torch.argmax(y_hat, dim=1)
#
#             # Move tensors back to CPU for visualization
#             img_np = img.cpu().numpy()[0, 0]  # Only show one image from the batch, slice out the first channel
#             mask_np = mask.cpu().numpy()[0]   # Ground truth
#             pred_np = y_hat_softmax.cpu().numpy()[0]  # Model prediction
#
#             # Plot original image, ground truth mask, and predicted mask
#             axs[i, 0].imshow(img_np[img_np.shape[0] // 2, :, :], cmap='gray')  # Show a slice from the 3D volume
#             axs[i, 0].set_title('Image')
#             axs[i, 0].axis('off')
#
#             axs[i, 1].imshow(mask_np[mask_np.shape[0] // 2, :, :], cmap='gray')
#             axs[i, 1].set_title('Ground Truth')
#             axs[i, 1].axis('off')
#
#             axs[i, 2].imshow(pred_np[pred_np.shape[0] // 2, :, :], cmap='gray')
#             axs[i, 2].set_title('Prediction')
#             axs[i, 2].axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# visualize_predictions(model, val_loader)
#
#
