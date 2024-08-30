import { Text ,View,Button } from "react-native";

const Register =({navigation}) =>{
    return(
        <View style={{alignItems:'center',justifyContent:'center',top:300}}>
            <Text>This is Register Page</Text>
            <Button title="Login" onPress={()=>navigation.goBack()}></Button>
            <Button title="Register" onPress={()=> navigation.navigate('Maintab')}></Button>
        </View>      
        
    );
}

export default Register;