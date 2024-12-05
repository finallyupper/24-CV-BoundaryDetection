import matplotlib.pyplot as plt 


THRESHOLD = [0.047619, 0.095238, 0.142857, 0.190476, 0.238095, 0.285714, 0.333333, 0.380952, 0.428571, 0.476190, 0.523810, 0.571429, 0.619048, 0.666667, 0.714286, 0.761905, 0.809524, 0.857143, 0.904762, 0.952381]

# Canny 
CANNY_RECALL = [0.648014, 0.648110, 0.648062, 0.648110, 0.648050, 0.648062, 0.648074, 0.648110, 0.648110, 0.648098, 0.648038, 0.648098, 0.648050, 0.648074, 0.648133, 0.648050, 0.648086, 0.648133, 0.648110, 0.648086]
CANNY_PRECISION = [0.135941, 0.135859, 0.135951, 0.135834, 0.135839, 0.135915, 0.135854, 0.135895, 0.135839, 0.135895, 0.135874, 0.135915, 0.135874, 0.135849, 0.135905, 0.135864, 0.135905, 0.135885, 0.135849, 0.135859]
CANNY_F1SCORE = [0.224736, 0.224630, 0.224753, 0.224595, 0.224599, 0.224704, 0.224621, 0.224679, 0.224602, 0.224678, 0.224647, 0.224706, 0.224648, 0.224614, 0.224695, 0.224634, 0.224692, 0.224667, 0.224616, 0.224629]

# Sobel
SOBEL_RECALL = [0.648098, 0.648050, 0.648110, 0.648086, 0.648086, 0.648098, 0.648110, 0.648086, 0.648026, 0.648121, 0.648110, 0.648074, 0.648038, 0.648074, 0.648086, 0.648062, 0.648074, 0.648026, 0.648086, 0.648086]
SOBEL_PRECISION = [0.135885, 0.135920, 0.135910, 0.135936, 0.135936, 0.135946, 0.135844, 0.135818, 0.135920, 0.135905, 0.135895, 0.135936, 0.135869, 0.135936, 0.135941, 0.135828, 0.135966, 0.135849, 0.135926, 0.135966]
SOBEL_F1SCORE = [0.224665, 0.224711, 0.224700, 0.224734, 0.224734, 0.224748, 0.224609, 0.224573, 0.224709, 0.224694, 0.224679, 0.224733, 0.224640, 0.224733, 0.224741, 0.224586, 0.224775, 0.224611, 0.224720, 0.224776]






def main():
    # F1 Score
    plt.plot(THRESHOLD, CANNY_F1SCORE, label="canny", color="green") 
    plt.plot(THRESHOLD, SOBEL_F1SCORE, label="sobel", color="orange")  
    plt.legend() 
    plt.xlabel("Threshold") 
    plt.ylabel("F1 score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./assets/f1score.png") 
    plt.close()

    # Precision
    plt.plot(THRESHOLD, CANNY_PRECISION, label="canny", color="green") 
    plt.plot(THRESHOLD, SOBEL_PRECISION, label="sobel", color="orange")  
    plt.legend() 
    plt.xlabel("Threshold") 
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./assets/precision.png") 
    plt.close()


    # Recall
    plt.plot(THRESHOLD, CANNY_RECALL, label="canny", color="green") 
    plt.plot(THRESHOLD, SOBEL_RECALL, label="sobel", color="orange")  
    plt.legend() 
    plt.xlabel("Threshold") 
    plt.ylabel("Recall")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./assets/recall.png") 
    plt.close()


if __name__ == "__main__":
    main()