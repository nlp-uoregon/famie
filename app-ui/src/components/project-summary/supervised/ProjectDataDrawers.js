import React from 'react';
import RuleTable from './RuleTable';
import LabelTable from './LabelTable';
import PerformanceTable from './PerformanceTable';
import AppBar from '@material-ui/core/AppBar';
import Tabs from '@material-ui/core/Tabs';
import Tab from '@material-ui/core/Tab';
import CircularProgress from '@material-ui/core/CircularProgress';
import FileDownloadButton from '../common/FileDownloadButton';
import DownloadUnlabeledDataButton from '../common/DownloadUnlabeledDataButton';
import ExportRulesButton from './ExportRulesButton';


class ProjectDataDrawers extends React.Component{ 

    state = {
        tab: 0
    }

    setTabValue = (e, value) => {
        this.setState({tab: value});
    }

    render(){
        if(!this.props.loading){
            return (
                <div className={this.props.classes.project_content}>
                    <AppBar position="static">
                        <Tabs value={this.state.tab} onChange={this.setTabValue}>
                            <Tab label="Data Statistics" />
                        </Tabs>
                    </AppBar>
                    <div className={this.props.classes.offset} style={{minHeight: '30px'}} />
                    {this.state.tab == 0 && 
                        <LabelTable
                            docs={this.props.docs}
                            projectType={this.props.projectType}
                            classNames={this.props.classNames}
                            exploreLabelled={this.props.exploreLabelled}
                            classes={this.props.classes}
                        />
                    }
                    <FileDownloadButton
                        projectName={this.props.projectName}
                        projectType={this.props.projectType}
                    />
                    <DownloadUnlabeledDataButton
                        projectName={this.props.projectName}
                        projectType={this.props.projectType}
                    />
                    <ExportRulesButton projectName={this.props.projectName}/>
                </div>
            )
        }
        else{
            return (
                <div className={this.props.classes.progress}>
                    <h3>Selecting best unlabeled examples for the next annotation iteration...</h3>
                    <p></p>
                    <CircularProgress/>
                </div>
            )
        }
    }
}

export default ProjectDataDrawers;