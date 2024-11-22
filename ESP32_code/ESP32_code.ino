int x;
int z=0;
int a=0;
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(1);
  pinMode(23,OUTPUT);
  pinMode(22,OUTPUT);
   pinMode(19,OUTPUT);
  pinMode(4,OUTPUT);
}

void loop() {
 
 
  while (!Serial.available());
  char x = Serial.read();
  if(x=='a'){
     z++;
  }
  else{
    z=0;
  }
  if(z>5){
    digitalWrite(23,HIGH);
    a=1;
    z=0;
  }
  if(x=='b'&&a==1){
    a=0;
    digitalWrite(23,LOW);
    digitalWrite(22,!digitalRead(22));
    delay(200);
  }
  if(x=='c'&&a==1){
    a=0;
    digitalWrite(23,LOW);
    digitalWrite(19,!digitalRead(19));
    delay(200);
  }
  if(x=='d'&&a==1){
    a=0;
    digitalWrite(23,LOW);
    digitalWrite(4,!digitalRead(4));
    delay(200);
  }
  delay(50);
  
}
