from pathlib import Path
from random import shuffle
from dir_mage import src_dir

if __name__ == '__main__':
    # indir_global = r'D:\paradise\stuff\simswappg\srcs'
    indir_global = r'D:\paradise\stuff\Essence\FS\all\Sluts'
    # targetDir_global = r'C:\Heaven\YummyBaker'
    # targetDir_global = r'C:\Heaven\YummyBaker'
    # targetDir_global = r'D:\paradise\stuff\new\imageset2\meri maa mujhse chud jati'
    # targetDir_global = 
    # targetDir_global = r'D:\paradise\stuff\new\pvd2'
    targetDir_global = r'D:\paradise\stuff\Essence\pictures\ranked2'
    
    # outDir_global = r'D:\paradise\stuff\new\pvd\test'
    outDir_global = r'D:\Developed\FaceSwapExperimental\TestResult'
    # outDir_global = r'C:\Games\sacred2'
    xtr = [x for x in Path(targetDir_global).glob('*') if x.is_dir()]
    shuffle(xtr)
    for dirt in xtr:
        trc = int(dirt.name)
        src_dir(indir_global,dirt,outDir_global,True,-1,trc)