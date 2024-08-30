import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import Setting from "./Setting";
import Store from "./Store";
import Chatbot from "./Chatbot";
import MaterialCommunityIcons from '@expo/vector-icons/MaterialCommunityIcons';
import Hometab from "./Home/Hometab";
const Tab =createBottomTabNavigator();

const Maintab =()=>{
    return(
        <Tab.Navigator
        screenOptions={({ route }) => ({
        tabBarActiveTintColor: 'tomato',
        tabBarInactiveTintColor: 'gray',
        tabBarIcon: ({ color, size }) => {
          let iconName;

          switch (route.name) {
            case 'Hometab':
              iconName = 'home';
              break;
            case 'Setting':
              iconName = 'account-outline';
              break;
            case 'Store':
              iconName = 'store';
              break;
            case 'Chatbot':
              iconName = 'chat';
              break;
            default:
              iconName = 'help';
              break;
          }

          return <MaterialCommunityIcons name={iconName} size={size} color={color} />;
        },
        tabBarStyle:{
          backgroundColor:'white'
        }
      })}
    >
        
            <Tab.Screen name="Hometab" component={Hometab} options={{headerShown:false}}></Tab.Screen>
            <Tab.Screen name="Chatbot" component={Chatbot}></Tab.Screen>
            <Tab.Screen name="Store" component={Store}></Tab.Screen>
            <Tab.Screen name="Setting" component={Setting}></Tab.Screen>
        </Tab.Navigator>
    );
}

export default Maintab;