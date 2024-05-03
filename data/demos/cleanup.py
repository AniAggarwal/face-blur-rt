import re

# Open the input CSV file and a new file for the corrected output
with open('./irl-stream-long-30fps-tinaface.csv', 'r') as infile, open('output.csv', 'w') as outfile:
    for line in infile:
        # Fix the bbox column by removing unnecessary quotes
        # This regex matches the quotes inside the outer quotes and replaces them
        fixed_line = re.sub(r'(?<=\[)"|"(?=\])|"(?=\[)|"(?=,")', '', line)
        
        # Write the fixed line to the output file
        outfile.write(fixed_line)

# Print done when processing is complete
print("Processing complete. Corrected file is 'output.csv'.")
