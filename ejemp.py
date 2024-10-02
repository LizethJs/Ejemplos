import requests

import json


class Capture:

    def __init__(self):  # Método inicializador de la clase.
        self.dataJson = []  # Inicializa un atributo dataJson como una lista vacía.
        self.finalJson = []


    def capturaDatos(self):  # Define un método llamado capturaDatos.
        response = requests.get("https://www.datos.gov.co/resource/m5pi-7cau.json")# Realiza una solicitud GET a la URL especificada y almacena la respuesta en la variable response.
        self.dataJson = response.json() # Convierte la respuesta a formato JSON y la almacena en dataJson.



    def Limpieza(self):  # Define un método llamado Limpieza.
        for ind in range(len(self.dataJson)):  # Itera sobre los índices de dataJson.
            jsonClean = {  # Inicializa un diccionario para almacenar datos limpios.
                "Year": '',  # Clave para el año, inicialmente vacía.
                "Quarter": '',  # Clave para el trimestre, inicialmente vacía.
                "Provider": '',  # Clave para el proveedor, inicialmente vacía.
                "Income": '',  # Clave para los ingresos, inicialmente vacía.
                "amountSMS": ''
                }
                # Realiza una serie de comprobaciones para asignar el nombre limpio del proveedor.
            if self.dataJson[ind]["proveedor"] == "ALMACENES EXITO INVERSIONES S.A.S.":
                jsonClean['Provider'] = "MOVIL EXITO"
            elif self.dataJson[ind]['proveedor'] == "AVANTEL S.A.S":
                jsonClean['Provider'] = "WOM"
            elif self.dataJson[ind]['proveedor'] == "VIRGIN MOBILE COLOMBIA S.A.S.":
                jsonClean['Provider'] = "VIRGIN MOBILE"
            elif self.dataJson[ind]['proveedor'] == "SUMA MOVIL S.A.S.":
                jsonClean['Provider'] = "SUMA MOVIL"  # Asigna un nombre limpio para el proveedor.
            elif self.dataJson[ind]['proveedor'] == "COLOMBIA MOVIL  S.A ESP":
                jsonClean['Provider'] = "CLARO"
            elif self.dataJson[ind]['proveedor'] == "LOGISTICA FLASH COLOMBIA S.A.S":
                jsonClean['Provider'] = "FLASH"
            elif self.dataJson[ind]['proveedor'] == "COMUNICACION CELULAR S A COMCEL S A":
                jsonClean['Provider'] = "CLARO"
            elif self.dataJson[ind]['proveedor'] == "EMPRESA DE TELECOMUNICACIONES DE BOGOTA S.A. ESP":
                jsonClean['Provider'] = "ETB"
            elif self.dataJson[ind]['proveedor'] == "SETROC MOBILE GROUP SAS":
                jsonClean['Provider'] = "SETROC"
            elif self.dataJson[ind]['proveedor'] == "PARTNERS TELECOM COLOMBIA SAS":
                jsonClean['Provider'] = "PARTNERS"
            elif self.dataJson[ind]['proveedor'] == "LIWA SAS ESP":
                jsonClean['Provider'] = "LIWA"
            elif self.dataJson[ind]['proveedor'] == "LOV TELECOMUNICACIONES SAS":
                jsonClean['Provider'] = "LOV"
            elif self.dataJson[ind]['proveedor'] == "UFF MOVIL SAS":
                jsonClean['Provider'] = "UFF"
            elif self.dataJson[ind]['proveedor'] == "COLOMBIA TELECOMUNICACIONES S.A. E.S.P.":
                jsonClean['Provider'] = "MOVISTAR"
            # Asigna el año, trimestre e ingresos del JSON original a los campos limpios.
            jsonClean['Year'] = self.dataJson[ind]['anno']
            jsonClean['Quarter'] = self.dataJson[ind]['trimestre']
            jsonClean['Income'] = self.dataJson[ind]['ingresos_por_mensajes']
            jsonClean['amountSMS'] = self.dataJson[ind]['cantidad_de_mensajes']
            self.finalJson.append(jsonClean)
        return self.finalJson
