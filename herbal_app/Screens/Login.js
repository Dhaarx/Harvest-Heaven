import { Text ,View,Button } from "react-native";
import Register from "./Register";

const Login =({navigation}) =>{
    return(
        <View style={{top:300}}>
            <Text>This is login Page</Text>
            <Button title="Login" onPress={()=>navigation.replace('Maintab')}></Button>
            <Button title="Register" onPress={()=>navigation.navigate('Register')}></Button>
        </View>      
        
    );
}

export default Login;