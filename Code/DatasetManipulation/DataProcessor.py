import csv

class DataProcessor:

    def __init__(self, path):
        self.outFile = open('output0.csv', 'w')
        self.writer = csv.writer(self.outFile)
        self.file = open('/home/millana/Desktop/ML/ML1819--task-101--team-23/Code/DatasetManipulation/output1.csv', 'r')
        self.reader = csv.reader(self.file, lineterminator='\n')
        # self.data = list(self.reader)
        print("Hellow")

    # Fills empty spaces from a column with zeros
    def fillBlanks(self, columnID, value):
        print("Filling blanks with zeros...\n")
        for row in self.reader:
            if row[columnID] in (None, ''):
                row[columnID] = 0
            self.writer.writerow(row)

    # Removes blank columns
    def removeBlankRow(self, columnID):
        print('Deleting rows that contain an empty cell at column '+str(columnID)+'...')
        for row in self.reader:
            if row[columnID] not in (None, ''):
                self.writer.writerow(row)
        

    # Converts column to binary output given a delimiter
    def toBinary(self, columnID, delimiter):
        print('Converting column '+str(columnID)+' to binary output using delimiter '+str(delimiter))
        for row in self.reader:
            if float(row[columnID]) > delimiter:
                self.writer.writerow(row+[1])
            else:
                self.writer.writerow(row+[0])

    def groupBy(self, columnID):
        print("Hello world!")

    def close(self):
        self.outFile.close()



if __name__ == '__main__':
    # processor = DataProcessor('../../Data/Test.csv')
    processor = DataProcessor('output.csv')

    # Fill blanks with zeros
    # processor.fillBlanks(11, '0')

    # processor.removeBlankRow(3)

    processor.toBinary(3, 0)

    processor.close()