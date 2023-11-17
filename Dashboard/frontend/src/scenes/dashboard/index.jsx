import { Box, Button, IconButton, Typography, useTheme } from "@mui/material";
import { tokens } from "../../theme";
import { mockTransactions } from "../../data/mockData";
import DownloadOutlinedIcon from "@mui/icons-material/DownloadOutlined";
import EmailIcon from "@mui/icons-material/Email";
import PointOfSaleIcon from "@mui/icons-material/PointOfSale";
import PersonAddIcon from "@mui/icons-material/PersonAdd";
import TrafficIcon from "@mui/icons-material/Traffic";
import Header from "../../components/Header";
import LineChart from "../../components/LineChart";
import GeographyChart from "../../components/GeographyChart";
import BarChart from "../../components/BarChart";
import StatBox from "../../components/StatBox";
import ProgressCircle from "../../components/ProgressCircle";

const Dashboard = ({setfile}) => {
  const theme = useTheme();
  const colors = tokens(theme.palette.mode);
  const handleFileChange = (e) => {
    setfile(e.target.files[0]);
  };

  return (
    <>
    <h1 style={{marginLeft: "20px"}}>Dashboard</h1>
    <Box m="20px">
      <Header title="Data Mining" subtitle="" />
      <h2>Upload CSV File</h2>
      <Button
        variant="contained"
        style={{color: "#757ce8", backgroundColor: "#e0e0e0", marginLeft: "30px"}}
        component="label"
      >
        <b>Upload File</b>
        <input
          onChange={handleFileChange}
          type="file"
          hidden
        />
      </Button>
      
    </Box>
    </>
  );
};

export default Dashboard;
