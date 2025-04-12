from django import forms
IPL_TEAM_CHOICES = [
    ("Chennai Super Kings", "Chennai Super Kings"),
    ("Delhi Capitals", "Delhi Capitals"),
    ("Gujarat Titans", "Gujarat Titans"),
    ("Kolkata Knight Riders", "Kolkata Knight Riders"),
    ("Lucknow Super Giants", "Lucknow Super Giants"),
    ("Mumbai Indians", "Mumbai Indians"),
    ("Punjab Kings", "Punjab Kings"),
    ("Rajasthan Royals", "Rajasthan Royals"),
    ("Royal Challengers Bengaluru", "Royal Challengers Bengaluru"),
    ("Sunrisers Hyderabad", "Sunrisers Hyderabad"),
]
IPL_VENUE_CHOICES = [
    ("MA Chidambaram Stadium, Chepauk, Chennai", "MA Chidambaram Stadium, Chepauk, Chennai"),
    ("Wankhede Stadium, Mumbai", "Wankhede Stadium, Mumbai"),
    ("Eden Gardens, Kolkata", "Eden Gardens, Kolkata"),
    ("Arun Jaitley Stadium, Delhi", "Arun Jaitley Stadium, Delhi"),
    ("Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam", "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam"),
    ("M.Chinnaswamy Stadium, Bengaluru", "M.Chinnaswamy Stadium, Bengaluru"),
    ("Rajiv Gandhi International Stadium, Uppal, Hyderabad", "Rajiv Gandhi International Stadium, Uppal, Hyderabad"),
    ("Narendra Modi Stadium, Motera, Ahmedabad", "Narendra Modi Stadium, Motera, Ahmedabad"),
    ("Sawai Mansingh Stadium, Jaipur", "Sawai Mansingh Stadium, Jaipur"),
    ("Barsapara Cricket Stadium, Guwahati", "Barsapara Cricket Stadium, Guwahati"),
    ("Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh", "Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh"),
    ("Himachal Pradesh Cricket Association Stadium, Dharamsala", "Himachal Pradesh Cricket Association Stadium, Dharamsala"),
    ("Maharaja Yadavindra Singh Stadium, Mullanpur", "Maharaja Yadavindra Singh Stadium, Mullanpur"),
    ("Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow", "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow"),
]
class TeamForm(forms.Form):
    team1 = forms.ChoiceField(choices=IPL_TEAM_CHOICES)
    team2 = forms.ChoiceField(choices=IPL_TEAM_CHOICES)
