import * as React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { createStackNavigator } from '@react-navigation/stack';
import { StyleSheet} from 'react-native';
import Login from './Screens/Login';
import Register from './Screens/Register';
import Maintab from './Screens/Maintab';

const Stack = createStackNavigator();

export default function App() {
  return (
   <NavigationContainer>
    <Stack.Navigator initialRouteName='login'>
      <Stack.Screen name="Login" component={Login}></Stack.Screen>
      <Stack.Screen name="Register" component={Register} options={{headerShown:false}}></Stack.Screen>
      <Stack.Screen name="Maintab" component={Maintab} options={{headerShown:false}}></Stack.Screen>
    </Stack.Navigator>
   </NavigationContainer>
  );
}


