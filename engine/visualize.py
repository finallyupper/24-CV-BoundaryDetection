import matplotlib.pyplot as plt 

THRESHOLD = [0.047619, 0.095238, 0.142857, 0.190476, 0.238095, 0.285714, 0.333333, 0.380952, 0.428571, 0.476190, 0.523810, 0.571429, 0.619048, 0.666667, 0.714286, 0.761905, 0.809524, 0.857143, 0.904762, 0.952381]

def load_data(file_path):
    recall = []
    precision = []
    f1score = []

    with open(file_path, "r") as file:
        lines = file.readlines()

    start_parsing = False
    for line in lines:
        if "[Results Per Threshold]" in line:
            start_parsing = True
            continue  
        
        if start_parsing and "Threshold" in line:
            continue 
        if start_parsing and not "Threshold" in line:
            if line.strip() == "" or not line[0].isdigit():
                break      
            parts = line.split()
            recall.append(float(parts[1]))
            precision.append(float(parts[2]))
            f1score.append(float(parts[3]))
    return recall, precision, f1score  


def main():
    canny = "../logs/canny_train.txt"
    sobel = "../logs/sobel_train.txt" 

    canny_recall, canny_precision, canny_f1 = load_data(canny) 
    sobel_recall, sobel_precision, sobel_f1 = load_data(sobel) 

    # F1 Score
    plt.plot(THRESHOLD, canny_f1, label="canny", color="green") 
    plt.plot(THRESHOLD, sobel_f1, label="sobel", color="orange")  
    plt.legend() 
    plt.xlabel("Threshold") 
    plt.ylabel("F1 score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../assets/f1score.png") 
    plt.close()

    # Precision
    plt.plot(THRESHOLD, canny_precision, label="canny", color="green") 
    plt.plot(THRESHOLD, sobel_precision, label="sobel", color="orange")  
    plt.legend() 
    plt.xlabel("Threshold") 
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../assets/precision.png") 
    plt.close()


    # Recall
    plt.plot(THRESHOLD, canny_recall, label="canny", color="green") 
    plt.plot(THRESHOLD, sobel_recall, label="sobel", color="orange")  
    plt.legend() 
    plt.xlabel("Threshold") 
    plt.ylabel("Recall")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../assets/recall.png") 
    plt.close()


if __name__ == "__main__":
    main()