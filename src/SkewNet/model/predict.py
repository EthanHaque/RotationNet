import argparse
import torch
import torchvision.transforms as transforms
from SkewNet.model.rotation_net import ModelRegistry
from SkewNet.model.train import SnapshotConfig
import math


def load_model(snapshot_path, device="cuda"):
    model = ModelRegistry.get_model("MobileNetV3LargeManualTest")  # Replace with actual model name
    model.load_state_dict(torch.load(snapshot_path).model_state)
    model.eval()
    model.to(device)
    return model


def image_transform(image):
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(image)


def predict_image(model, image):
    batch = create_image_batch([image])

    with torch.no_grad():
        prediction = model(batch)
    return prediction.item()


def batch_predict_images(model, images, batch_size=32):
    no_batches = math.ceil(len(images) / batch_size)

    predictions = []
    for i in range(no_batches):
        start = i * batch_size
        end = (i + 1) * batch_size
        batch = create_image_batch(images[start:end])

        with torch.no_grad():
            network_output = model(batch)

        network_output_list = network_output.tolist()
        for output in network_output_list:
            predictions.append(output[0])

    return predictions


def create_image_batch(images, device="cuda"):
    batch = []
    for image in images:
        image_tensor = image_transform(image)
        batch.append(image_tensor)
    batch = torch.stack(batch)
    batch = batch.to(device)
    return batch


def main(snapshot_path, image_path):
    model = load_model(snapshot_path)
    prediction = predict_image(model, image_path)
    print(f"Prediction: {prediction}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Image Prediction Using a Pretrained Model")
    parser.add_argument("snapshot_path", help="Path to the model snapshot")
    parser.add_argument("image_path", help="Path to the input image")
    args = parser.parse_args()

    main(args.snapshot_path, args.image_path)
