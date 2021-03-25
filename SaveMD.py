import RNG as RN_Gomoku
import sys

if(len(sys.argv)<2):
    a0eng=RN_Gomoku.A0_ENG(64,"./RNG64.tf",1e-1/(2.0**3),10)
    a0eng.a0_eng.save("RNG")
    a0eng2=RN_Gomoku.A0_ENG(64,"./weights/RNG64.tf",1e-1/(2.0**3),10)
    a0eng2.a0_eng.save("RNG_Old")
    quit()

if(sys.argv[1]=="n"):
    a0eng=RN_Gomoku.A0_ENG(64,"./RNG64.tf",1e-1/(2.0**3),10)
    a0eng.a0_eng.save("RNG")
elif(sys.argv[1]=="o"):
    a0eng2=RN_Gomoku.A0_ENG(64,"./weights/RNG64.tf",1e-1/(2.0**3),10)
    a0eng2.a0_eng.save("RNG_Old")
else:
    print("Error")