from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import sys
import logging
from werkzeug.serving import run_simple
import webbrowser
from threading import Timer

if getattr(sys, 'frozen', False):
    template_folder = os.path.join(sys._MEIPASS, 'templates')
    static_folder = os.path.join(sys._MEIPASS, 'static')
    checkpoint_dir = os.path.dirname(sys.executable)
    app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
    app.debug = True
else:
    checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
    app = Flask(__name__)

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

# Import necessary SAM modules
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Create Flask application
app = Flask(__name__, template_folder='files/templates', static_folder='files/static')

# Configure upload folder
UPLOAD_FOLDER = 'files/static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Get the absolute path to the checkpoint file
#checkpoint_dir = os.path.dirname(os.path.abspath(__file__))
#sam_checkpoint = os.path.join(checkpoint_dir, 'files', 'static', 'sam_vit_h_4b8939.pth')
checkpoint_rel_path = os.path.join('files', 'static', 'sam_vit_h_4b8939.pth')
sam_checkpoint = os.path.join(checkpoint_dir, checkpoint_rel_path)

# Load the SAM model
#sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda" if torch.cuda.is_available() else "cpu"
#print(device)

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

# Define the image segmentation route
@app.route('/segment', methods=['POST'])
def segment_image():
    # Check if an image was uploaded
    if 'image' not in request.files:
        return render_template('index.html', error='No image uploaded')
    
    # Get the uploaded image file
    image_file = request.files['image']
    
    # Validate the file extension
    if image_file.filename == '':
        return render_template('index.html', error='No image selected')
    if not allowed_file(image_file.filename):
        return render_template('index.html', error='Invalid file extension')

    # Save the image to the upload folder
    filename = secure_filename(image_file.filename)
    #image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_path  = os.path.join('files/static/uploads/', filename)
    image_file.save(image_path)

    # Get segmentation parameters from the form
    points_per_side = int(request.form.get('pointsPerSide'))
    pred_iou_thresh = float(request.form.get('predIouThresh'))
    stability_score_thresh = float(request.form.get('stabilityScoreThresh'))
    crop_n_layers = int(request.form.get('cropNLayers'))
    crop_n_points_downscale_factor = int(request.form.get('cropNPointsDownscaleFactor'))
    min_mask_region_area = int(request.form.get('minMaskRegionArea'))
    output_image_dpi = int(request.form.get('outputImageDpi'))
    #output_image_size_height = int(request.form.get('outputImageSizeHeight'))
    #output_image_size_breadth = int(request.form.get('outputImageSizeBreadth'))

    # Perform image segmentation
    masks = perform_image_segmentation(
        image_path,
        points_per_side,
        pred_iou_thresh,
        stability_score_thresh,
        crop_n_layers,
        crop_n_points_downscale_factor,
        min_mask_region_area
    )
    
    # Generate visualization of the segmentation masks
    visualization_path = generate_segmentation_visualization(
        image_path,
        masks,
        output_image_dpi
        #output_image_size_height,
        #output_image_size_breadth
    )
    
    image_path = os.path.relpath(image_path, "files/")
    visualization_path = os.path.relpath(visualization_path, "files/")

    # Display the visualization in the result page
    return render_template('result.html', image_path=image_path, visualization_path=visualization_path)

# Helper function to check allowed file extensions
def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to perform image segmentation
def perform_image_segmentation(
    image_path,
    points_per_side,
    pred_iou_thresh,
    stability_score_thresh,
    crop_n_layers,
    crop_n_points_downscale_factor,
    min_mask_region_area
):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        crop_n_layers=crop_n_layers,
        crop_n_points_downscale_factor=crop_n_points_downscale_factor,
        min_mask_region_area=min_mask_region_area
    )

    masks = mask_generator.generate(image)

    return masks

# Helper function to generate visualization of segmentation masks
'''def generate_segmentation_visualization(
    image_path,
    masks,
    output_image_dpi,
    output_image_size_height,
    output_image_size_breadth
):
    image = Image.open(image_path)
    canvas = np.zeros_like(image)

    for ann in masks:
        m = ann['segmentation']
        color_mask = np.random.rand(1, 1, 3) * 255
        canvas[np.where(m)] = color_mask

    fig, ax = plt.subplots(figsize=(output_image_size_height/output_image_dpi, output_image_size_breadth/output_image_dpi), dpi=output_image_dpi)
    ax.imshow(canvas.astype(np.uint8))
    ax.axis('off')
    plt.tight_layout(pad=0)

    # Save the image to a file
    image_name = os.path.basename(image_path)
    visualization_filename = f'{image_name}_visualization.png'
    visualization_path = os.path.join(app.config['UPLOAD_FOLDER'], visualization_filename)
    plt.savefig(visualization_path, bbox_inches='tight', pad_inches=0, dpi=output_image_dpi)
    plt.close()

    return visualization_path'''

def generate_segmentation_visualization(
    image_path,
    masks,
    output_image_dpi,
):
    image = Image.open(image_path)
    image_size = image.size
    canvas = np.zeros_like(image)

    for ann in masks:
        m = ann['segmentation']
        color_mask = np.random.rand(1, 1, 3) * 255
        canvas[np.where(m)] = color_mask

    fig, ax = plt.subplots(figsize=(image_size[0] / output_image_dpi, image_size[1] / output_image_dpi), dpi=output_image_dpi)
    ax.imshow(canvas.astype(np.uint8))
    ax.axis('off')
    plt.tight_layout(pad=0)

    # Save the image to a file
    image_name = os.path.basename(image_path)
    visualization_filename = f'{image_name}_visualization.png'
    visualization_path = os.path.join(app.config['UPLOAD_FOLDER'], visualization_filename)
    plt.savefig(visualization_path, bbox_inches='tight', pad_inches=0, dpi=output_image_dpi)
    plt.close()

    return visualization_path



if __name__ == '__main__':
    #Timer(1, open_browser).start()
    if getattr(sys, 'frozen', False):
        # Running as an executable (PyInstaller)
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    #app.run(server='flask', threaded=True, use_reloader=False)
    #run_simple('192.168.22.19', 5000, app, threaded=True, use_reloader=False)
    app.run(host='192.168.22.11')