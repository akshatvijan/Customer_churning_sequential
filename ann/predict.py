import torch


def predict_churn_probability(model, features, device):

    model.eval()

    features = torch.tensor(
        features,
        dtype=torch.float32
    ).to(device)

    with torch.no_grad():

        logits = model(features)

        probability = torch.sigmoid(
            logits
        )

    return probability.cpu().numpy()