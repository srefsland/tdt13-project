import argparse
import os


# Converts fine grained tags to coarse grained tags

fine_grained_to_coarse = {'O': 'O', 'B-Facility': 'B-LOC', 'I-Facility': 'I-LOC', 'B-OtherLOC': 'B-LOC', 'I-OtherLOC': 'I-LOC',
                          'B-HumanSettlement': 'B-LOC', 'I-HumanSettlement': 'I-LOC', 'B-Station': 'B-LOC', 'I-Station': 'I-LOC',
                          'B-VisualWork': 'B-CW', 'I-VisualWork': 'I-CW', 'B-MusicalWork': 'B-CW', 'I-MusicalWork': 'I-CW',
                          'B-WrittenWork': 'B-CW', 'I-WrittenWork': 'I-CW', 'B-ArtWork': 'B-CW', 'I-ArtWork': 'I-CW',
                          'B-Software': 'B-CW', 'I-Software': 'I-CW', 'B-OtherCW': 'B-CW', 'I-OtherCW': 'I-CW',
                          'B-MusicalGRP': 'B-GRP', 'I-MusicalGRP': 'I-GRP', 'B-PublicCorp': 'B-GRP', 'I-PublicCorp': 'I-GRP',
                          'B-PrivateCorp': 'B-GRP', 'I-PrivateCorp': 'I-GRP', 'B-OtherCorp': 'B-GRP', 'I-OtherCorp': 'I-GRP',
                          'B-AerospaceManufacturer': 'B-GRP', 'I-AerospaceManufacturer': 'I-GRP', 'B-SportsGRP': 'B-GRP', 'I-SportsGRP': 'I-GRP',
                          'B-CarManufacturer': 'B-GRP', 'I-CarManufacturer': 'I-GRP', 'B-TechCorp': 'B-GRP', 'I-TechCorp': 'I-GRP',
                          'B-ORG': 'B-GRP', 'I-ORG': 'I-GRP', 'B-Scientist': 'B-PER', 'I-Scientist': 'I-PER',
                          'B-Artist': 'B-PER', 'I-Artist': 'I-PER', 'B-Athlete': 'B-PER', 'I-Athlete': 'I-PER',
                          'B-Politician': 'B-PER', 'I-Politician': 'I-PER', 'B-Cleric': 'B-PER', 'I-Cleric': 'I-PER',
                          'B-SportsManager': 'B-PER', 'I-SportsManager': 'I-PER', 'B-OtherPER': 'B-PER', 'I-OtherPER': 'I-PER',
                          'B-Clothing': 'B-PROD', 'I-Clothing': 'I-PROD', 'B-Vehicle': 'B-PROD', 'I-Vehicle': 'I-PROD',
                          'B-Food': 'B-PROD', 'I-Food': 'I-PROD', 'B-Drink': 'B-PROD', 'I-Drink': 'I-PROD',
                          'B-OtherPROD': 'B-PROD', 'I-OtherPROD': 'I-PROD', 'B-Medication/Vaccine': 'B-MED', 'I-Medication/Vaccine': 'I-MED',
                          'B-MedicalProcedure': 'B-MED', 'I-MedicalProcedure': 'I-MED', 'B-AnatomicalStructure': 'B-MED', 'I-AnatomicalStructure': 'I-MED',
                          'B-Symptom': 'B-MED', 'I-Symptom': 'I-MED', 'B-Disease': 'B-MED', 'I-Disease': 'I-MED'}

def convert_to_coarse(text_file, prefix):
    # Read in the file
    with open(text_file, 'r') as file:
        lines = file.readlines()

    converted_lines = []

    for line in lines:
        if line != '\n' and not line.startswith('#'):
            # Use regex to split on multiple spaces
            tokens = line.split()
            tokens[-1] = fine_grained_to_coarse.get(tokens[-1], tokens[-1]) + '\n'
            line = ' '.join(tokens)
            converted_lines.append(line)
        else:
            converted_lines.append(line)

    directory = os.path.dirname(text_file)
    file = os.path.basename(text_file)
    # Write the file out again with double newlines between sections
    with open(f"{directory}/{prefix}{file}", 'w') as file:
        file.write("".join(converted_lines))
        print("Wrote to", prefix + text_file)
        
if __name__ == "__main__":
    # parse args
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--text_file', nargs='+', type=str, default='train.txt')
    arg_parser.add_argument('--prefix', type=str, default='coarse_')
    
    args = arg_parser.parse_args()
    text_files = args.text_file
    prefix = args.prefix
    
    for text_file in text_files:
        convert_to_coarse(text_file, prefix)
    
        