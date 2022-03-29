import React from 'react';
import $ from 'jquery';
import { Redirect } from 'react-router-dom';
import { withStyles } from '@material-ui/core/styles';
import SideBar from '../SideBar';
import SingleFileUploadForm from '../file-upload/SingleFileUploadForm';
import { DEFAULT_CLASS_NAME_COLUMN, DOCS_CLASSNAME_FILE_FORMAT } from  '../constants';
import Papa from 'papaparse';
import _ from 'lodash';


const styles = theme => ({
    container: {display: 'flex', flexDirection: 'row'},
    subcontainer: {display: 'flex', flexDirection: 'column'},
    top_container: {marginLeft: 10}
  });


  const initialiseSelectedColumns = () => {
    var columnNames = {};
    columnNames[DEFAULT_CLASS_NAME_COLUMN] = undefined;
    return columnNames;
}

class ClassNames extends React.Component{

    state = {
        toRules: false,
        loadingLabel: false,
        loadingData: false,
        selectedInputColumns: initialiseSelectedColumns(),
        candidateInputColumnNames: undefined,
        fileUploaded: undefined,
        dataUploaded: undefined
    };

    submitClassNames = (selectedFile, columnName) => {
        
        console.log('Inside submitClassNames');

        if(!selectedFile){
            alert("Need to select file!");
        }

        if(!this.props.projectName){
            alert('Do not know what project this is.')
        }

        const data = new FormData();
        data.append('project_name', this.props.projectName);
        data.append('column_name', columnName);
        data.append('file', selectedFile);
        console.log('data', ...data);

        $.ajax({
            url : '/api/classnames',
            type : 'POST',
            data : data,
            processData: false,  // tell jQuery not to process the data
            contentType: false,  // tell jQuery not to set contentType,
            success : function(data) {
                this.setState({ loadingLabel: false });
                const jsonData = JSON.parse(data);
                if(jsonData){
                    console.log("Classes set correctly to: ", jsonData);

                    this.setState( {loadingLabel: false,
                                    fileUploaded: selectedFile.name} );
                    this.props.setProjectParams(jsonData);
                }
            }.bind(this),
            error: function (error) {
                this.setState({ loadingLabel: false });
                alert(error);
            }.bind(this)
        });

        this.setState({ loadingLabel: true });
        
    }

    submitLabeledData = (selectedFile, columnName) => {

        console.log('Inside submitLabeledData');

        if(!selectedFile){
            alert("Need to select file!");
        }

        if(!this.props.projectName){
            alert('Do not know what project this is.')
        }

        const data = new FormData();
        data.append('project_name', this.props.projectName);
        data.append('column_name', columnName);
        data.append('file', selectedFile);
        console.log('data', ...data);

        $.ajax({
            url : '/api/upload-labeled-data',
            type : 'POST',
            data : data,
            processData: false,  // tell jQuery not to process the data
            contentType: false,  // tell jQuery not to set contentType,
            success : function(data) {
                this.setState({ loadingData: false });
                const jsonData = JSON.parse(data);
                if(jsonData){
                    this.setState( {loadingData: false,
                                    dataUploaded: selectedFile.name} );
                }
            }.bind(this),
            error: function (error) {
                this.setState({ loadingData: false });
                alert(error);
            }.bind(this)
        });

        this.setState({ loadingData: true });

    }

    updateCandidateInputColumnNames = (candidateInputColumnNames) => {
        this.setState({ candidateInputColumnNames });
    }

    updateSelectedInputColumns = (columnIndex) => {
        this.setState((prevState) => {
            const selectedInputColumns = {...prevState.selectedInputColumns};
            selectedInputColumns[DEFAULT_CLASS_NAME_COLUMN] = columnIndex;
            return { selectedInputColumns };
        })
    }

    validateColumnsAndUpload = (selectedFile, columns) => {
        console.log("Inside validateColumnsAndUpload", DEFAULT_CLASS_NAME_COLUMN,
        columns, _.includes(columns, DEFAULT_CLASS_NAME_COLUMN))
        this.submitClassNames(selectedFile, DEFAULT_CLASS_NAME_COLUMN);
    }

    validateColumnsAndUpload2 = (selectedFile, columns) => {
        console.log("Inside validateColumnsAndUpload2", DEFAULT_CLASS_NAME_COLUMN,
        columns, _.includes(columns, DEFAULT_CLASS_NAME_COLUMN))
        this.submitLabeledData(selectedFile, DEFAULT_CLASS_NAME_COLUMN);
    }

    uploadClassNamesFile = ({selectedFile}) => {
        console.log("Inside uploadClassNamesFile", this.props, selectedFile);

        const selectedColumn = this.state.selectedInputColumns[DEFAULT_CLASS_NAME_COLUMN];
        var results = Papa.parse(selectedFile,
                {header: true,
                preview: 1,
                complete: function(results) {
                        this.validateColumnsAndUpload(selectedFile, results.meta.fields);
                    }.bind(this)
        });
    }

    uploadLabeledData = ({selectedFile}) => {
        console.log("Inside uploadLabeledData", this.props, selectedFile);

        const selectedColumn = this.state.selectedInputColumns[DEFAULT_CLASS_NAME_COLUMN];
        var results = Papa.parse(selectedFile,
                {header: true,
                preview: 1,
                complete: function(results) {
                        this.validateColumnsAndUpload2(selectedFile, results.meta.fields);
                    }.bind(this)
        });
    }

    setToNextPage = () => {
        this.setState({toRules: true});
    }

    setToNextPage2 = () => {
        this.setState({toRules: false});
    }

    render() {
        const { classes } = this.props;

        if(this.state.toRules === true) {
            return <Redirect to={{pathname: "/projects/" + this.props.projectName}}/>
        }

        return (
            <div className={classes.container}>
                <SideBar/>
                <div className={"subcontainer"}>
                <SingleFileUploadForm 
                    id={"contained-button-file"}
                    helpText={<p>File is empty.</p>}
                    rootClassName={classes.top_container}
                    instructionText={"Load a plaintext file, each line contains one type for labeling."}
                    defaultColumnNames={[DEFAULT_CLASS_NAME_COLUMN]}
                    createProject={this.uploadClassNamesFile}
                    projectName={this.props.projectName}
                    loading={this.state.loadingLabel}
                    candidateInputColumnNames={this.state.candidateInputColumnNames}
                    updateSelectedInputColumns={this.updateSelectedInputColumns}
                    selectedInputColumns={this.state.selectedInputColumns}
                    fileUploaded={this.state.fileUploaded}
                    setToNextPage={this.setToNextPage}
                    setProjectUploadFinished={this.props.setProjectParamsFinished}
                    uploadButtonText={"Upload Label Set"}
                />

                <SingleFileUploadForm
                    id={"contained-button-file-2"}
                    helpText={<p>File is empty.</p>}
                    rootClassName={classes.top_container}
                    instructionText={"Load a BIO-format file."}
                    defaultColumnNames={[DEFAULT_CLASS_NAME_COLUMN]}
                    createProject={this.uploadLabeledData}
                    projectName={this.props.projectName}
                    loading={this.state.loadingData}
                    candidateInputColumnNames={this.state.candidateInputColumnNames}
                    updateSelectedInputColumns={this.updateSelectedInputColumns}
                    selectedInputColumns={this.state.selectedInputColumns}
                    fileUploaded={this.state.dataUploaded}
                    uploadButtonText={"Upload Labeled Data"}
                />
                </div>
            </div>
        )
    }
}

export default withStyles(styles)(ClassNames);