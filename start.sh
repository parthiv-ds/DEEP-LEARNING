echo "creating virtualenvironment"
virtualenv deeplear

echo "changing the directory"
cd deeplear/Scripts

echo "Activateing virtual environment"
source activate

echo "change to main directory"
cd ..
cd ..

echo "Installing requirements"
pip install -r requirements.txt 

echo "run main.py"
python main.py

#(:'':-for multiline comment,#:-single line comment)
#deactivate
