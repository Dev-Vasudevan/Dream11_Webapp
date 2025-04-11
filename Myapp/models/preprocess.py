from torch.utils.data import Dataset, DataLoader,random_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
import torch
import pandas as pd
class Data(Dataset):
    def __init__(self,xy,device):
        self.x = xy[0].to(device)
        self.y = xy[1].to(device)
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
class FantasyData(Dataset):
    def __init__(self,bat,bowl):

        self.batting , self.bowling = self.transform_data(self.join_data(bat,bowl))

    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    def join_data(self,batting,bowling):
        batting.drop(['bowling_team','batting_team','batting_innings' , 'batting_position'],axis=1,inplace=True)
        bowling.drop(['bowling_team','batting_team','bowling_innings' , 'total_balls', 'dots', 'maidens', 'conceded', 'foursConceded',
       'sixesConceded','wides', 'noballs', 'LBW',
       'Hitwicket', 'CaughtBowled', 'Bowled'],axis=1,inplace=True)
        return pd.merge(batting, bowling,  how='outer')
    def transform_data(self , df , bowling = True  , batting = True ):
        self.bowler_data = {}
        self.batter_data = {}
        df = df[df['season'] > 2015]

        batting_transformed_rows = []
        bowling_transformed_rows = []

        # Process each batter separately
        for player, group in df.groupby("fullName"):
            batting_prev = -1
            bowling_prev = -1
            # Sort the batter's matches in chronological order
            group = group.sort_values(["season", "match_id"]).reset_index(drop=True)
            # List to keep track of previous match scores for this batter
            batting_previous_scores = []
            bowling_previous_scores = []

            # Iterate over the batter's matches
            for i, row in group.iterrows():
                current_season = row["season"]
                num_matches = i

                if batting:
                    # Previous match score: last score in the list, if available
                    batting_prev_match_score = batting_previous_scores[-1] if batting_previous_scores else None

                    # Previous 5 matches average points: average of last 5 scores
                    last_five = batting_previous_scores[-5:] if batting_previous_scores else []
                    batting_prev_5_avg = sum(last_five) / len(last_five) if last_five else None

                    # Previous season average points: consider all matches of this batter with season < current_season
                    batting_prev_season_scores = group[group["season"] < current_season]["Batting_FP"].tolist()
                    batting_prev_season_avg = sum(batting_prev_season_scores) / len(batting_prev_season_scores) if batting_prev_season_scores else None

                    batting_row_dict = {
                    "Batter name": row["fullName"],
                    "batting_prev": batting_prev_match_score,
                    "batting_prev5": batting_prev_5_avg,
                    "batting_prevSSN": batting_prev_season_avg,
                    "num matches": num_matches,
                    "venue": row["venue"],
                    "season": current_season,
                    "Batting_FP" : row["Batting_FP"],
                    }
                # row_dict.extend(venue_encoding)

                    batting_transformed_rows.append(batting_row_dict)
                    batting_prev = batting_row_dict
                    batting_previous_scores.append(row["Batting_FP"])
                # Bowling Data


                if bowling:
                    bowling_prev_match_fp = bowling_previous_scores[-1] if bowling_previous_scores else None

                    # Previous 5 matches average Bowling_FP: average of last 5 scores
                    last_five = bowling_previous_scores[-5:] if bowling_previous_scores else []
                    bowling_prev_5_avg_fp = sum(last_five) / len(last_five) if last_five else None
                    num_matches = len(bowling_previous_scores)

                    # Previous season average Bowling_FP: consider all matches of this bowler with season < current_season
                    bowling_prev_season_scores = group[group["season"] < current_season]["Bowling_FP"].tolist()
                    bowling_prev_season_avg_fp = sum(bowling_prev_season_scores) / len(bowling_prev_season_scores) if bowling_prev_season_scores else None







                # Create the dictionary for the current row


                    bowling_row_dict = {
                        "Bowler name": row["fullName"],
                        "venue": row["venue"],
                        "bowling_prev": bowling_prev_match_fp,
                        "bowling_prev5": bowling_prev_5_avg_fp,
                        "bowling_prevSSN": bowling_prev_season_avg_fp,
                        "season": current_season,
                        "num matches": num_matches,
                        "Bowling_FP": row["Bowling_FP"],

                    }

                    bowling_transformed_rows.append(bowling_row_dict)
                    bowling_prev = bowling_row_dict

                    # Update the list of previous scores with the current match's Bowling_FP
                    bowling_previous_scores.append(row["Bowling_FP"])
            self.batter_data[player] = batting_prev
            self.bowler_data[player] = bowling_prev


        return pd.DataFrame(batting_transformed_rows),pd.DataFrame(bowling_transformed_rows)


    def get_batting_data(self,name,venue , season ):
        if name not in self.batter_data:
            self.batter_data[name] = {'Batter name': name,
             'batting_prev': 0,
             'batting_prev5': 0,
             'batting_prevSSN': 0,
             'num matches': 0}
        data = self.batter_data[name].copy()
        data["venue"] = [venue]
        data["season"] = [season]
        data = pd.DataFrame(data)
        data["Batting_FP"] = [0]*len(data)
        return self.preprocess_batting(data,False)

    def get_bowling_data(self,name,venue , season ):
        if name not in self.bowler_data:
            self.bowler_data[name]= {'Bowler name': name,
             'bowling_prev': 0,
             'bowling_prev5': 0,
             'bowling_prevSSN': 0,
             'num matches': 0}

        data = self.bowler_data[name].copy()
        data["venue"] = [venue]
        data["season"] = [season]
        data = pd.DataFrame(data)
        data["Bowling_FP"] = [0]*len(data)
        return self.preprocess_bowling(data,False)
    # Assumes transformed data
    def preprocess_batting(self, df , train = True  ):
        # Targets
        y =  torch.tensor(
            df[["Batting_FP"]].fillna(0).values,
            dtype=torch.float32
        )

        # Compute sums for each column
        positive_sums = torch.sum(y * (y > 0), dim=0)  # Sum of positive values
        negative_sums = torch.sum(-y * (y < 0), dim=0)  # Sum of negative values

        # Avoid division by zero by adding a small epsilon
        epsilon = 1e-8
        scale_factors = positive_sums / (negative_sums + epsilon)
        scale_factors = scale_factors.unsqueeze(0) * 0.05  # Shape: (1, 4)

        # Apply scaling only to negative values (correct broadcasting)
        y = torch.where(y < 0, y * scale_factors, y)
        #print(y.shape)






        # 2. One-hot encode teams
        if train :
            self.venue_enc = OneHotEncoder(sparse_output=False)
            venue = self.venue_enc.fit_transform(df[['venue']])
            #print(self.venue_enc)
        else :

            #print([[df['venue']]])
            venue = self.venue_enc.transform(df[['venue']])
            #print(venue)


        # 3. Process season
        if train :
            self.scaler = MinMaxScaler()
            self.scaler.fit([[2015],[2025]])
        season_scaled = self.scaler.transform(df[['season']])

        # Convert to tensors and combine
        season_tensor = torch.tensor(season_scaled, dtype=torch.float32)
        venue_tensor = torch.tensor(venue, dtype=torch.float32)

        batting_prev = torch.tensor(df[["batting_prev"]].fillna(0).values, dtype=torch.float32)
        batting_prev5 = torch.tensor(df[["batting_prev5"]].fillna(0).values, dtype=torch.float32)
        batting_prevSSN = torch.tensor(df[["batting_prevSSN"]].fillna(0).values, dtype=torch.float32)
        if train:
            self.scaler_matches = MinMaxScaler()
            matches_scaled = self.scaler_matches.fit_transform(df[['num matches']].fillna(0))
        else:
            matches_scaled = self.scaler_matches.transform(df[['num matches']])

        num_matches = torch.tensor(matches_scaled, dtype=torch.float32)

        x = torch.cat([
            batting_prev,
            batting_prev5,
            batting_prevSSN,
            num_matches,
            season_tensor,
            venue_tensor,
        ], dim=1)
        # #print(x.shape)

        return x,y
    def preprocess_bowling(self, df,train = True):


        # Targets (Bowling_FP)
        y = torch.tensor(
            df[["Bowling_FP"]].fillna(0).values,
            dtype=torch.float32
        )

        # Compute sums for scaling negative values
        positive_sums = torch.sum(y * (y > 0), dim=0)  # Sum of positive values
        negative_sums = torch.sum(-y * (y < 0), dim=0)  # Sum of negative values

        epsilon = 1e-8  # Avoid division by zero
        scale_factors = positive_sums / (negative_sums + epsilon)
        scale_factors = scale_factors.unsqueeze(0) * 0.05

        y = torch.where(y < 0, y * scale_factors, y)

        # One-hot encode venues
        if train:
            self.venue_enc = OneHotEncoder(sparse_output=False)
            venue_encoded = self.venue_enc.fit_transform(df[['venue']])
        else:
            venue_encoded = self.venue_enc.transform(df[['venue']])

        venue_tensor = torch.tensor(venue_encoded, dtype=torch.float32)

        # Scale seasons using MinMaxScaler
        if train:
            self.scaler = MinMaxScaler()
            self.scaler.fit([[2015],[2025]])
        season_scaled = self.scaler.transform(df[['season']])

        season_tensor = torch.tensor(season_scaled, dtype=torch.float32)


        bowling_prev_tensor = torch.tensor(df[["bowling_prev"]].fillna(0).values, dtype=torch.float32)
        bowling_prev5_tensor = torch.tensor(df[["bowling_prev5"]].fillna(0).values, dtype=torch.float32)
        bowling_prevSSN_tensor = torch.tensor(df[["bowling_prevSSN"]].fillna(0).values, dtype=torch.float32)
        if train:
            self.scaler_matches = MinMaxScaler()
            matches_scaled = self.scaler_matches.fit_transform(df[['num matches']].fillna(0))
        else:
            matches_scaled = self.scaler_matches.transform(df[['num matches']])

        season_tensor = torch.tensor(season_scaled, dtype=torch.float32)
        num_matches = torch.tensor(matches_scaled, dtype=torch.float32)

        # Combine all features into a single tensor (x)
        x = torch.cat([

            bowling_prev_tensor,
            bowling_prev5_tensor,
            bowling_prevSSN_tensor,
            num_matches,
            season_tensor,
            venue_tensor,
        ], dim=1)

        return x, y

def train(model, train_data_loader, optimizer, loss_fn):
    total_loss = 0.0
    model.train()  # Set the model to training mode

    for data in train_data_loader:
        inputs, labels = data  # Unpack inputs and labels
        inputs = inputs.detach()
        labels = labels.detach()
        optimizer.zero_grad()  # Clear gradients from previous step

        outputs = model(inputs)  # Forward pass
        loss = loss_fn(outputs, labels)  # Compute loss

        loss.backward()  # Backward pass (no need for retain_graph=True)
        optimizer.step()  # Update weights

        total_loss += loss.item()  # Accumulate loss

    # Compute average loss over all batches
    avg_loss = total_loss / len(train_data_loader)

    return avg_loss
def evaluate(model, eval_data_loader ):
    model.eval()  # Set model to evaluation mode
    total_error = 0  # Assuming 4 output columns
    total_samples = 0

    with torch.no_grad():  # Disable gradient computation
        for data in eval_data_loader:
            inputs, labels = data
            inputs = inputs.detach()
            labels = labels.detach()

            # Forward pass
            outputs = model(inputs)

            # Calculate squared error for each sample and each output
            error = (outputs - labels)**2

            # Sum errors across batches
            total_error += error.sum(dim=0)
            total_samples += labels.size(0)

    # Calculate average error for each column
    avg_error = total_error / total_samples

    return torch.sqrt(avg_error)

