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
    ("M. A. Chidambaram Stadium, Chennai", "M. A. Chidambaram Stadium, Chennai"),
    ("Wankhede Stadium, Mumbai", "Wankhede Stadium, Mumbai"),
    ("Eden Gardens, Kolkata", "Eden Gardens, Kolkata"),
    ("Arun Jaitley Stadium, Delhi", "Arun Jaitley Stadium, Delhi"),
    ("ACA–VDCA Cricket Stadium, Visakhapatnam", "ACA–VDCA Cricket Stadium, Visakhapatnam"),
    ("M. Chinnaswamy Stadium, Bengaluru", "M. Chinnaswamy Stadium, Bengaluru"),
    ("Rajiv Gandhi International Stadium, Hyderabad", "Rajiv Gandhi International Stadium, Hyderabad"),
    ("Narendra Modi Stadium, Ahmedabad", "Narendra Modi Stadium, Ahmedabad"),
    ("Sawai Mansingh Stadium, Jaipur", "Sawai Mansingh Stadium, Jaipur"),
    ("Barsapara Cricket Stadium, Guwahati", "Barsapara Cricket Stadium, Guwahati"),
    ("PCA Stadium, Mohali", "PCA Stadium, Mohali"),
    ("HPCA Stadium, Dharamsala", "HPCA Stadium, Dharamsala"),
    ("Maharaja Yadavindra Singh Stadium, Mullanpur", "Maharaja Yadavindra Singh Stadium, Mullanpur"),
    ("Ekana Cricket Stadium, Lucknow", "Ekana Cricket Stadium, Lucknow"),
]
class TeamForm(forms.Form):
    team1 = forms.ChoiceField(choices=IPL_TEAM_CHOICES)
    team2 = forms.ChoiceField(choices=IPL_TEAM_CHOICES)
    venue = forms.ChoiceField(choices=IPL_VENUE_CHOICES)
