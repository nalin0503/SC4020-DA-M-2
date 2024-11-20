import torch
import torch.nn as nn
from RNN import sequence_to_vectors_from_city

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, lengths):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(device)
        # Pack the padded sequence
        x_packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        # Pass through RNN
        out_packed, _ = self.rnn(x_packed, h0)
        # Unpack the sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        # Gather the outputs at the last valid time step
        idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, out.size(2)).to(device)
        out = out.gather(1, idx).squeeze(1)
        # Pass through fully connected layer
        out = self.fc(out)
        # Apply log softmax
        out = nn.functional.log_softmax(out, dim=1)
        return out

model_path = 'rnn_model.pth'


# Hyperparameters
input_size = 768
hidden_size = 128
output_size = 85
num_layers = 2
learning_rate = 0.0005
num_epochs = 30

location_categories = """
Food
Shopping
Entertainment
Japanese restaurant
Western restaurant
Eat all you can restaurant
Chinese restaurant
Indian restaurant
Ramen restaurant
Curry restaurant
BBQ restaurant
Hot pot restaurant
Bar
Diner
Creative cuisine
Organic cuisine
Pizza
Café
Tea Salon
Bakery
Sweets 
Wine Bar
Pub
Disco
Beer Garden
Fast Food
Karaoke
Cruising
Theme Park Restaurant
Amusement Restaurant
Other Restaurants
Glasses
Drug Store
Electronics Store
DIY Store
Convenience Store
Recycle Shop
Interior Shop
Sports Store
Clothes Store
Grocery Store
Online Grocery Store
Sports Recreation
Game Arcade
Swimming Pool
Hotel
Park
Transit Station
Parking Area
Casino
Hospital
Pharmacy
Chiropractic
Elderly Care Home
Fishing
School
Cram School
Kindergarten
Real Estate
Home Appliances
Post Office
Laundry 
Driving School
Wedding Ceremony
Cemetary
Bank
Vet
Hot Spring
Hair Salon
Lawyer Office
Recruitment Office
City Hall
Community Center
Church
Retail Store
Accountant Office
IT Office
Publisher Office
Building Material
Gardening
Heavy Industry
NPO
Utility Copany
Port
Research Facility
"""

location_categories = location_categories.split('\n')[1:-1]

if __name__ == '__main__':
    
    # Initialize the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleRNN(input_size, hidden_size, output_size, num_layers).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    sequence = []
    # Get the input sequence from the user
    while True:
        tup = input("Enter the sequence of coordinates of where you have been before (separated by commas): ")
        
        try:
            first = int(tup.split(',')[0])
            second = int(tup.split(',')[1])
            sequence.append((first, second))
        except:
            print("Invalid input. Please try again.")
            continue
        
        cont = input("Do you want to add more coordinates? (y/n): ")
        if cont == 'n':
            break
            

    city = 'A'
    X = sequence_to_vectors_from_city(sequence, city)
    # Put X in a minibatch of size 1
    X = X[0].unsqueeze_(0)
    
    lengths = torch.tensor([len(X)])
    
    output = model(X, lengths)
    
    # Get the top 5 categories
    output = torch.argsort(output, dim=1, descending=True).squeeze(0).tolist()[:5]
    output = [location_categories[i] for i in output]

    print("\nThe top 5 categories of places you might be interested in are:")
    print(output)


