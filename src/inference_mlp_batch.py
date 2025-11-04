# features: predict on 10 random test samples (combined)
import torch, numpy as np
from src.data_loader_features import get_feature_loaders
from src.model_mlp import MLPClassifier
from src.config import DEVICE

activity = ["WALKING","WALKING_UPSTAIRS","WALKING_DOWNSTAIRS","SITTING","STANDING","LAYING"]
train_loader, test_loader, n_classes = get_feature_loaders("combined", batch_size=1)
input_dim = next(iter(train_loader))[0].shape[1]
model = MLPClassifier(input_dim=input_dim, n_classes=n_classes, dropout=0.0).to(DEVICE)
model.load_state_dict(torch.load("./saved_models/best_mlp_combined.pth", map_location=DEVICE))
model.eval()
avg = 0
count = 0
for i, (X, y) in zip(range(1000), test_loader):
    if i >= 1000:
        break
    with torch.no_grad():
        logits = model(X.to(DEVICE).float())
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    pred = probs.argmax()
    avg += probs[pred]
    count += 1
    print(f"{i}: true={activity[int(y.item())]:<18} pred={activity[pred]:<18} conf={probs[pred]:.4f}")
print(f"Average confidence: {avg/count:.4f}")