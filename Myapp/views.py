import torch
import pickle
import dill
from django.http import JsonResponse
from django.shortcuts import render
from .models.model import Model
from Myapp.models.preprocess import FantasyData
import pandas as pd
from .forms import  *
# Load the trained PyTorch model
batting_model_path = 'Myapp/models/batting_new.pth'  # Update with the actual path
bowling_model_path = 'Myapp/models/bowling_new.pth'  # Update with the actual path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batting_model =Model()
bowling_model =Model()
batting_model.load_state_dict(torch.load(batting_model_path, map_location=device))
bowling_model.load_state_dict(torch.load(batting_model_path, map_location=device))

batting_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize FantasyData preprocessor
with open('Myapp/models/fantasy_data.pkl', 'rb') as f:
    preprocessor = dill.load(f)
    

ipl_2025_squads = {
    "Mumbai Indians": [
        "Hardik Pandya", "Jasprit Bumrah", "Rohit Sharma", "Tilak Varma", "Trent Boult", "Naman Dhir", "Robin Minz",
        "Karn Sharma", "Ryan Rickelton", "Deepak Chahar", "Allah Ghazanfar", "Will Jacks", "Ashwani Kumar",
        "Mitchell Santner", "Reece Topley", "Shrijith Krishnan", "Raj Angad Bawa", "Venkat Satyanarayana Raju",
        "Bevon Jacobs", "Arjun Tendulkar", "Lizaad Williams", "Vignesh Puthhur", "Suryakumar Yadav"
    ],
    "Punjab Kings": [
        "Shreyas Iyer", "Yuzvendra Chahal", "Arshdeep Singh", "Marcus Stoinis", "Glenn Maxwell", "Shashank Singh",
        "Prabhsimran Singh", "Harpreet Brar", "Vijaykumar Vyshak", "Yash Thakur", "Marco Jansen", "Josh Inglis",
        "Lockie Ferguson", "Azmatullah Omarzai", "Harnoor Pannu", "Kuldeep Sen", "Priyansh Arya", "Aaron Hardie",
        "Musheer Khan", "Suryansh Shedge", "Xavier Bartlett", "Pyla Avinash", "Pravin Dubey", "Nehal Wadhera"
    ],
    "Rajasthan Royals": [
        "Sanju Samson", "Yashasvi Jaiswal", "Riyan Parag", "Dhruv Jurel", "Shimron Hetmyer", "Sandeep Sharma",
        "Jofra Archer", "Wanindu Hasaranga", "Maheesh Theekshana", "Akash Madhwal", "Kumar Kartikeya Singh",
        "Nitish Rana", "Tushar Deshpande", "Shubham Dubey", "Yudhvir Charak", "Fazalhaq Farooqi", "Vaibhav Suryavanshi",
        "Kwena Maphaka", "Kunal Rathore", "Ashok Sharma"
    ],
    "Royal Challengers Bengaluru": [
        "Virat Kohli", "Rajat Patidar", "Yash Dayal", "Josh Hazlewood", "Phil Salt", "Jitesh Sharma",
        "Liam Livingstone",
        "Rasikh Dar", "Suyash Sharma", "Krunal Pandya", "Bhuvneshwar Kumar", "Swapnil Singh", "Tim David",
        "Romario Shepherd", "Nuwan Thushara", "Manoj Bhandage", "Jacob Bethell", "Devdutt Padikkal", "Swastik Chhikara",
        "Lungi Ngidi", "Abhinandan Singh", "Mohit Rathee"
    ],
    "Sunrisers Hyderabad": [
        "Pat Cummins", "Travis Head", "Abhishek Sharma", "Heinrich Klaasen", "Nitish Reddy", "Ishan Kishan",
        "Mohammed Shami", "Harshal Patel", "Rahul Chahar", "Adam Zampa", "Atharva Taide", "Abhinav Manohar",
        "Simarjeet Singh", "Zeeshan Ansari", "Jaydev Unadkat", "Brydon Carse", "Kamindu Mendis", "Aniket Verma",
        "Eshan Malinga", "Sachin Baby"
    ],
    "Chennai Super Kings": [
        "Ruturaj Gaikwad", "MS Dhoni", "Ravindra Jadeja", "Shivam Dube", "Matheesha Pathirana", "Noor Ahmad",
        "Ravichandran Ashwin", "Devon Conway", "Khaleel Ahmed", "Rachin Ravindra", "Rahul Tripathi", "Vijay Shankar",
        "Sam Curran", "Shaik Rasheed", "Anshul Kamboj", "Mukesh Choudhary", "Deepak Hooda", "Gurjapneet Singh",
        "Nathan Ellis", "Jamie Overton", "Kamlesh Nagarkoti", "Ramakrishna Ghosh", "Shreyas Gopal", "Vansh Bedi",
        "Andre Siddarth"
    ],
    "Delhi Capitals": [
        "KL Rahul", "Harry Brook", "Jake Fraser-McGurk", "Karun Nair", "Abhishek Porel", "Tristan Stubbs", "Axar Patel",
        "Kuldeep Yadav", "T Natarajan", "Mitchell Starc", "Sameer Rizvi", "Ashutosh Sharma", "Mohit Sharma",
        "Faf du Plessis", "Mukesh Kumar", "Darshan Nalkande", "Vipraj Nigam", "Dushmantha Chameera", "Donovan Ferreira",
        "Ajay Mandal", "Manvanth Kumar", "Tripurana Vijay", "Madhav Tiwari"
    ],
    "Gujarat Titans": [
        "Shubman Gill", "Jos Buttler", "B. Sai Sudharsan", "Shahrukh Khan", "Kagiso Rabada", "Mohammed Siraj",
        "Prasidh Krishna", "Rahul Tewatia", "Rashid Khan", "Nishant Sindhu", "Mahipal Lomror", "Kumar Kushagra",
        "Anuj Rawat", "Manav Suthar", "Washington Sundar", "Gerald Coetzee", "Mohammad Arshad Khan",
        "Gurnoor Singh Brar", "Sherfane Rutherford", "R. Sai Kishore", "Ishant Sharma", "Jayant Yadav",
        "Glenn Phillips", "Karim Janat", "Kulwant Khejroliya"
    ],
    "Lucknow Super Giants": [
        "Rishabh Pant", "David Miller", "Aiden Markram", "Nicholas Pooran", "Mitchell Marsh", "Avesh Khan",
        "Mayank Yadav", "Mohsin Khan", "Ravi Bishnoi", "Abdul Samad", "Aryan Juyal", "Akash Deep", "Himmat Singh",
        "M Siddharth", "Digvesh Singh", "Shahbaz Ahmed", "Akash Singh", "Shamar Joseph", "Prince Yadav",
        "Yuvraj Chaudhary", "Rajvardhan Hangargekar", "Arshin Kulkarni", "Matthew Breetzke"
    ],
    "Kolkata Knight Riders": [
        "Venkatesh Iyer", "Rinku Singh", "Quinton de Kock", "Rahmanullah Gurbaz", "Angkrish Raghuvanshi",
        "Rovman Powell", "Manish Pandey", "Ajinkya Rahane", "Anukul Roy", "Moeen Ali", "Ramandeep Singh",
        "Andre Russell", "Anrich Nortje", "Vaibhav Arora", "Mayank Markande", "Spencer Johnson", "Umran Malik",
        "Harshit Rana", "Sunil Narine", "Varun Chakravarthy", "Luvnith Sisodia"
    ]
}

def predict_fantasy_points(request):
    if request.method == 'POST':
        form = TeamForm(request.POST)
        if form.is_valid():
            # Get inputs from the POST request (sent from the webpage)
            player_names = [player for player in ipl_2025_squads[form.cleaned_data['team1']]+ipl_2025_squads[form.cleaned_data['team2']]]
            venue = form.cleaned_data['venue']
            season = 2025
            top_totals = []
            # Preprocess the inputs
            for player in player_names:
                batting_x, _ = preprocessor.get_batting_data(player, venue, season)
                bowling_x, _ = preprocessor.get_bowling_data(player, venue, season)

                with torch.no_grad():
                    batting_x = batting_x.to(device)
                    bowling_x = bowling_x.to(device)

                    # Predict batting and bowling points
                    batting_points = batting_model(batting_x).item()
                    bowling_points = bowling_model(bowling_x).item()
                top_totals.append((player,batting_points+bowling_points))
            top_totals.sort(key=lambda x: x[1],reverse=True)


            # Return predictions as JSON
            return render(request,"predicton.html" , {"players" : top_totals[:11]})
    else :
        form = TeamForm()
    # Render a simple form to take inputs from the user
    return render(request, 'team_selector.html', {"form":form})

