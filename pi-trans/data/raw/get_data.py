from subprocess import run



data_type = ["Train", "Test"]
freq = ["Daily", "Weekly", "Monthly", "Quarterly", "Yearly", "Hourly"]

for i in data_type:
    for j in freq:
        run(
            [
                "wget",
                f"https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/{i}/{j}-{i.lower()}.csv"
            ]
        )
        
run(
    ["wget", "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/M4-info.csv"]
)