import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2
import rasterio
from pyproj import Proj, transform

# Load the SAM2 model
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Device setup
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

class Segment:
    def __init__(self):
        # Initialize the main Tkinter window
        self.root = tk.Tk()
        self.root.title("Segmentation Tool")

        # Dynamically set paths relative to this script's location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        image_path = os.path.join(script_dir, '../images/ortho.jpg')
        tif_path = os.path.join(script_dir, '../images/ortho.tif')
        sam2_checkpoint = os.path.join(script_dir, '../deps/sam2/checkpoints/sam2.1_hiera_large.pt')
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        # Load the image
        image = Image.open(image_path)
        self.image = np.array(image.convert("RGB"))
        self.dataset = rasterio.open(tif_path)

        # Build the model
        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        self.predictor.set_image(image)

        self.mask_input = None  # low resolution image for next prediction
        self.input_points = []
        self.input_labels = []
        self.pos_scatters = []
        self.neg_scatters = []
        self.old_masks = []

        # Matplotlib Figure and Canvas for embedding in Tkinter
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Show initial image
        self.ax.imshow(self.image)
        self.ax.set_title("Left Click = Contained (1), Right Click = Not Contained (0)")
        self.canvas.draw()

        # Connect matplotlib events to handle clicks
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

        # Initialize variables to control behavior
        self.mode = "segment"  # Default mode

        # Add a Button in the Tkinter GUI for segmentation mode
        self.segmentation_button = tk.Button(self.root, text="Start Segmentation", command=self.start_segmentation)
        self.segmentation_button.pack(side=tk.RIGHT, padx=10, pady=10)

        self.pixel_button = tk.Button(self.root, text="Get Pixel Location", command=self.enable_pixel_location)
        self.pixel_button.pack(side=tk.RIGHT, padx=10, pady=10)

    def start_segmentation(self):
        self.mode = "segment"
        print("Segmentation mode activated.")
        self.root.after(10, self.root.update_idletasks())  # update GUI after interaction

    def enable_pixel_location(self):
        self.mode = "pixel_location"
        print("Pixel location mode activated.")
        self.root.after(10, self.root.update_idletasks())

    @staticmethod
    def show_mask(mask, ax, random_color=False, borders=True):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        if borders:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(contour, epsilon=0.001, closed=True) for contour in contours]
            largest_contour = max(contours, key=cv2.contourArea)
            mask_image = cv2.drawContours(mask_image, [largest_contour], -1, (255, 0, 0, 0.5), thickness=30)

        mask = ax.imshow(mask_image, animated=True)
        return mask

    def onclick(self, event):
        if event.xdata is None or event.ydata is None:
            return  # Ignore clicks outside the plot area

        x, y = int(event.xdata), int(event.ydata)

        # Left click: contained (label = 1), Right click: not contained (label = 0)
        if self.mode == "segment":
            if event.button == 1:  # Left click
                self.input_points.append([x, y])
                self.input_labels.append(1)
            elif event.button == 3:  # Right click
                self.input_points.append([x, y])
                self.input_labels.append(0)
            elif event.button == 2:  # Middle click
                self.clear_old_points_and_save_masks()
                self.input_points = []
                self.input_labels = []
                self.pos_scatters = []
                self.neg_scatters = []
                self.mask_input = None
                return

            # Make a prediction
            input_point = np.array(self.input_points)
            input_label = np.array(self.input_labels)

            pos_scatter, neg_scatter = self.show_points(np.array(input_point), np.array(input_label), self.ax)
            self.pos_scatters.append(pos_scatter)
            self.neg_scatters.append(neg_scatter)

            if self.mask_input is not None:
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    mask_input=self.mask_input[None, :, :],
                    multimask_output=False,
                )
            else:
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=False
                )

            self.mask_input = logits[np.argmax(scores), :, :]

            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]

            # Show the best mask
            self.show_masks(self.image, masks, scores, point_coords=input_point, input_labels=input_label)

        elif self.mode == "pixel_location":
            print(f"Pixel location clicked: ({x}, {y})")
            # Code to handle getting the pixel location
            self.show_pixel_location(x, y)

    def show_pixel_location(self, x, y):
        # Display pixel location on the image
        self.ax.scatter([x], [y], color='red', marker='x', s=100)
        gps_coords = self.dataset.transform * (x, y)
        utm_proj = Proj(proj="utm", zone=18, ellps="WGS84", south=False)
        lat, lon = utm_proj(gps_coords[0], gps_coords[1], inverse=True)
        print(f"(Longitude: {lon}, Latitude: {lat})")
        self.ax.text(x + 30, y + 10, f"({lon:.6f}, {lat:.6f})", color='blue', fontsize=12, weight='bold')
        self.canvas.draw()

    def clear_old_masks(self):
        for mask in self.old_masks:
            mask.remove()
        self.old_masks = []
        self.canvas.draw_idle()

    def clear_old_points_and_save_masks(self):
        for point in self.pos_scatters:
            point.remove()
        for point in self.neg_scatters:
            point.remove()
        self.old_masks = []
        self.canvas.draw_idle()

    def show_masks(self, image, masks, scores, point_coords=None, input_labels=None, borders=True):
        # Clear the axes and reset them
        # self.ax.clear()
        # self.ax.imshow(image)
        # self.ax.set_title("Left Click = Contained (1), Right Click = Not Contained (0)")
        self.clear_old_masks()

        # Draw the masks on the cleared image
        for i, (mask, score) in enumerate(zip(masks, scores)):
            old_mask = self.show_mask(mask, self.ax, borders=borders)
            self.old_masks.append(old_mask)
            # if point_coords is not None:
            #     assert input_labels is not None

        # Force a redraw of the canvas to show updates
        self.canvas.draw()

    def show_points(self, coords, labels, ax, marker_size=375):
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        pos_scatter = ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*',
                                 s=marker_size, edgecolor='white', linewidth=1.25)
        neg_scatter = ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*',
                                 s=marker_size, edgecolor='white', linewidth=1.25)
        return pos_scatter, neg_scatter

    def start(self):
        self.root.mainloop()  # Start the Tkinter event loop


if __name__ == "__main__":
    segmenter = Segment()
    segmenter.start()
