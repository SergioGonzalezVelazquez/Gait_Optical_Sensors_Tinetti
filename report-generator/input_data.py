from __future__ import print_function, unicode_literals

from PyInquirer import style_from_dict, Token, prompt, Separator
from prompt_toolkit.validation import Validator, ValidationError
from pprint import pprint
import easygui
from bcolors import bcolors


class NumberValidator(Validator):
    def validate(self, document):
        try:
            int(document.text)
        except ValueError:
            raise ValidationError(message="Por favor, introduce un número entero",
                                  cursor_position=len(document.text))
style = style_from_dict({
    Token.Separator: '#cc5454',
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '',
})


general_questions = [
    {
        'type': 'input',
        'name': 'name',
        'message': 'Nombre',
    },
    {
        'type': 'input',
        'name': 'age',
        'message': 'Edad',
        'validate': NumberValidator,
    },
    {
        'type': 'list',
        'name': 'gender',
        'message': 'Género',
        'choices': ["Hombre", "Mujer", "No especificado", ]
    },
    {
        'type': 'input',
        'name': 'height',
        'message': 'Altura (en centímetros)',
        'validate': NumberValidator,
    },
    {
        'type': 'input',
        'name': 'weight',
        'message': 'Peso (en kilogramos)',
        'validate': NumberValidator,
    },
    {
        'type': 'input',
        'name': 'foot_length',
        'message': 'Longitud de pie (en milímetros)',
        'validate': NumberValidator,
    },
]

predictors_type_questions = [
    {
        'type': 'list',
        'name': 'predictors_type',
        'message': 'Tipo de parámetros de la marcha utilizados para generar automáticamente las respuetas a la prueba de Tinetti',
        'choices': ["Espaciotemporales", "Cinematicas", "Espaciotemporales y cinematicas", ]
    },
]

# https://medium.com/geekculture/build-interactive-cli-tools-in-python-47303c50d75
# https://codeburst.io/building-beautiful-command-line-interfaces-with-python-26c7e1bb54df


def input_data():
    print("Características generales del paciente")
    answers = prompt(general_questions, style=style)
    answers['height'] = int(answers['height'])
    answers['foot_length'] = int(answers['foot_length'])
    answers['weight'] = int(answers['weight'])
    print("\nConfiguración para la generación de informes")
    answers.update(prompt(predictors_type_questions, style=style))
    predictors_type = answers['predictors_type'].lower().replace(" ", "_")
    if predictors_type == 'espaciotemporales':
        answers['predictors_type'] = 'st'
    elif predictors_type == 'cinematicas':
        answers['predictors_type'] = 'kin'
    else:
        answers['predictors_type'] = 'stkin'
    subject = "subject_05"
    record = "rec_06"
    print("Seleccione el archivo con la trayectoria de los marcadores (exportado desde Clinical 3DMA)")
    answers['raw_file'] = easygui.fileopenbox(filetypes=['*.txt'])
    print(bcolors.OKGREEN + answers['raw_file'] + bcolors.ENDC)

    print("Seleccione el archivo de eventos de la marcha (exportado desde Clinical 3DMA)")
    answers['events_file'] = easygui.fileopenbox(filetypes=['*.txt'])
    print(bcolors.OKGREEN + answers['events_file'] + bcolors.ENDC)

    print("Seleccione el archivo con el desplazamiento del centro de gravedad (exportado desde Clinical 3DMA)")
    answers['cog_file'] = easygui.fileopenbox(filetypes=['*.csv'])
    print(bcolors.OKGREEN + answers['cog_file'] + bcolors.ENDC)

    print("Seleccione el directorio con los archivos CSV de datos cinemáticos (exportados desde el diálogo de zancadas de Clinical 3DMA)")
    answers['kin_path'] = easygui.diropenbox()
    print(bcolors.OKGREEN + answers['kin_path'] + bcolors.ENDC)
    '''
    subject = "subject_02"
    record = "rec_01"
    answers = {'age': 23,
               'foot_length': 270,
               'leg_length': 1000,
               'gender': 'Hombre',
               'height': 170,
               'name': 'Sergio',
               'predictors_type': 'stkin',
               'events_file': 'D:/dataset/optitrack/' + subject +  '/' + record + '/' + subject + "_" + record + '.events.TXT',
               'cog_file': 'D:/dataset/optitrack/' + subject +  '/' + record + '/biomechanics/COG.csv',
               'raw_file':  'D:/dataset/optitrack/' + subject +  '/' + record + '/' + subject + "_" + record + '.raw',
               'kin_path': 'D:/dataset/optitrack/' + subject +  '/' + record + '/strides/',
               'weight': 69}
    '''
    return answers
