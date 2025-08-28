import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';

// Import our screens
import AssetExample from './components/AssetExample';
import FoodDetailScreen from './components/FoodDetailScreen';
import FoodLensScreen from './components/FoodLensScreen';
import GetStartedScreen from './components/GetStartedScreen';
import GetStartedScreen1 from './components/GetStartedScreen1';
import GetStartedScreen3 from './components/GetStartedScreen3';
import ImageuploadingScreen from './components/ImageuploadingScreen';
import imageuploadScreen1 from './components/imageuploadScreen1';
import LoginScreen from './components/LoginScreen';
import profileScreen from './components/profileScreen';
import recipeScreen from './components/recipeScreen';
import RegisterScreen from './components/RegisterScreen';
import UploadedScreen from './components/UploadedScreen';

const Stack = createStackNavigator();

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator
        initialRouteName="GetStarted"
        screenOptions={{
          headerShown: false, // Hide the header for all screens
        }}
      >
        <Stack.Screen name="GetStarted" component={GetStartedScreen} />
        <Stack.Screen name="GetStarted1" component={GetStartedScreen1} />
        <Stack.Screen name="GetStarted3" component={GetStartedScreen3} />
        <Stack.Screen name="Register" component={RegisterScreen} />
        <Stack.Screen name="Login" component={LoginScreen} />
        
        <Stack.Screen name="FoodLens" component={FoodLensScreen} />
        <Stack.Screen name="FoodDetail" component={FoodDetailScreen} />
        <Stack.Screen name="ImageUploading" component={ImageuploadingScreen} />
        <Stack.Screen name="ImageUpload1" component={imageuploadScreen1} />
        <Stack.Screen name="Uploaded" component={UploadedScreen} />
        <Stack.Screen name="Profile" component={profileScreen} />
        <Stack.Screen name="Recipe" component={recipeScreen} />
        <Stack.Screen name="AssetExample" component={AssetExample} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}