import time
import copy
import torch
import matplotlib.pyplot as plt

def train_model(model, criterion, optimizer,data_loaders, dataset_sizes, num_epochs=25): #scheduler
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    epoch_losses = []
    epoch_accuracies = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs-1}")
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in data_loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad() # zero the parameter gradients

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            # if phase == 'train':
            #     scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}")

            epoch_losses.append(epoch_loss)
            epoch_accuracies.append(epoch_acc)
            plt.plot(epoch_losses, list(range(epoch)))


            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
        print()
    
    time_elapased = time.time() - since
    print(f'Training complete in {time_elapased // 60:.0f}m {time_elapased % 60:.0f}s')
    print(f'Best val accuracy: {best_acc:4f}')

    plt.show()

    # Use the best model
    model.load_state_dict(best_model_wts)
    return model